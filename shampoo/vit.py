import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from typing import Dict, Iterator, List, Optional, Tuple
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import functools
# Hugging Face datasets 라이브러리 import
from datasets import load_dataset
from PIL import Image

# [변경] timm 라이브러리 import
try:
    from timm.data import create_transform, Mixup
except ImportError:
    print("ERROR: timm library not found. Please install it using 'pip install timm'")
    exit(1)

# 제공된 Shampoo 옵티마이저 라이브러리 import

from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    AdaGradGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    GraftingConfig,
    RMSpropGraftingConfig,
    RWSAdaGradGraftingConfig,
    SGDGraftingConfig,
)

# --- ViT 모델 코드 (변경 없음) ---
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536, attn_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        x = x + attn_output
        norm_x_mlp = self.norm2(x)
        mlp_output = self.mlp(norm_x_mlp)
        x = x + mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, num_classes: int = 1000, embedding_dim: int = 384, depth: int = 12, num_heads: int = 6, mlp_dim: int = 1536, attn_dropout: float = 0.0, mlp_dropout: float = 0.1, embedding_dropout: float = 0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, attn_dropout=attn_dropout, mlp_dropout=mlp_dropout) for _ in range(depth)])
        self.classifier_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=num_classes)
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x).flatten(2, 3).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.classifier_norm(x)
        cls_token_final = x[:, 0]
        logits = self.classifier_head(cls_token_final)
        return logits

# --- 학습률 스케줄러 (변경 없음) ---
def get_warmup_cosine_decay_lr(current_step: int, base_lr: float, num_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (num_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * cosine_decay

# --- 분산 학습 설정 ---
def setup():
    # torchrun이 설정한 env:// 사용
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
def cleanup():
    dist.destroy_process_group()

# [변경] transform 타입을 명시적으로 지정하지 않고 callable 객체를 받도록 수정 (timm 호환성)
def apply_transforms(examples: Dict[str, List[Image.Image]], transform) -> Dict[str, List[torch.Tensor]]:
    # timm transform은 PIL 이미지를 입력으로 받습니다. RGB 변환 확인
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
    return examples
        
# --- 학습 로직 ---
def train(args: argparse.Namespace):
    setup()
    # 글로벌 랭크(전체 프로세스 ID)와 로컬 랭크(노드 내 GPU ID)
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    print(f"Running DDP training. Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}")

    # TensorBoard 로거는 rank 0에서만 생성합니다.
    writer = SummaryWriter(log_dir=args.log_dir) if global_rank == 0 else None

    # [변경] --- 데이터 증강 및 Mixup 설정 (timm 사용) ---
    
    # 1. RandAugment (timm.data.create_transform 사용)
    # Algoperf ViT Spec: RandAugment(layers=2, magnitude=15) -> 'rand-m15-n2-mstd0.5'
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        auto_augment='rand-m15-n2-mstd0.5',
        interpolation='bicubic', # ViT는 보통 bicubic 사용
        # mean/std는 ImageNet 기본값이 사용됩니다.
    )

    val_transform = create_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
    )

    # 2. Mixup (timm.data.Mixup 사용)
    # Algoperf ViT Spec: Mixup(alpha=0.2), Label Smoothing(0.1)
    mixup_fn = None
    if args.mixup > 0 or args.label_smoothing > 0:
        mixup_args = {
            'mixup_alpha': args.mixup,
            'cutmix_alpha': 0.0,  # CutMix 사용 안함
            'label_smoothing': args.label_smoothing,
            'num_classes': 1000
        }
        # mixup_fn은 학습 루프 내에서 배치 단위로 적용됩니다.
        mixup_fn = Mixup(**mixup_args)

    # --- 데이터셋 및 데이터로더 ---
    # rank 0에서만 데이터셋을 다운로드하고, 다른 프로세스는 기다립니다.
    if global_rank == 0:
        print("Hugging Face Hub에서 ImageNet-1k 데이터셋을 다운로드 및 캐싱합니다...")
        load_dataset("imagenet-1k", cache_dir=args.data_path)

    dist.barrier()

    print(f"Rank {global_rank}에서 캐시된 ImageNet-1k 데이터셋을 로딩합니다...")
    dataset = load_dataset("imagenet-1k", cache_dir=args.data_path)

    train_dataset = dataset['train']
    train_dataset.set_transform(functools.partial(apply_transforms, transform=train_transform))
    
    val_dataset = dataset['validation']
    val_dataset.set_transform(functools.partial(apply_transforms, transform=val_transform))
    
    def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'label': torch.tensor([x['label'] for x in batch], dtype=torch.long)
        }

    # Sampler는 global_rank를 사용합니다.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    
    # 검증 데이터로더도 DistributedSampler를 사용하여 데이터를 분할해야 합니다.
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)


    # --- 모델, 손실 함수, 최적화기 ---
    vit_s16_params = {
        'img_size': 224, 'patch_size': 16, 'embedding_dim': 384, 'depth': 12,
        'num_heads': 6, 'mlp_dim': 1536, 'num_classes': 1000
    }
    # 모델은 local_rank (해당 노드의 GPU ID)에 할당합니다.
    model = VisionTransformer(**vit_s16_params).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # [변경] 손실 함수 설정
    # Mixup/Label Smoothing이 적용된 소프트 레이블을 처리하기 위해 표준 CrossEntropyLoss 사용 (PyTorch 1.10+)
    # 이전 코드의 label_smoothing=0.1은 Mixup 객체에서 처리되므로 제거합니다.
    criterion = nn.CrossEntropyLoss().to(local_rank)

    
    # [변경] Shampoo 옵티마이저 설정: 하이퍼파라미터를 args에서 가져옵니다.
    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, 0.99),
        epsilon=1e-8,
        momentum=args.beta1,      # Shampoo의 momentum은 일반적으로 beta1과 동일하게 설정
        weight_decay=args.weight_decay,
        max_preconditioner_dim=1024,
        precondition_frequency=100,
        use_normalized_grafting=False,
        inv_root_override=2,
        exponent_multiplier=1,
        start_preconditioning_step=args.warmup_steps+1,
        use_nadam=False,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=0.99, epsilon=1e-8),
        distributed_config=DDPShampooConfig()
    )
    start_epoch = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            # 분산 학습 환경에서는 master process(rank 0)만 로드하고 다른 프로세스에 broadcast 하거나,
            # 각 프로세스가 동일한 파일을 로드하게 할 수 있습니다. 후자가 더 간단합니다.
            # 체크포인트는 항상 CPU에 먼저 로드하는 것이 안전합니다.
            checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')

            # 현재 저장된 체크포인트는 모델 가중치만 포함하고 있습니다.
            if 'model_state_dict' in checkpoint:
                 model.module.load_state_dict(checkpoint['model_state_dict'])
            else: # 이전 방식 호환 (model.module.state_dict()만 저장한 경우)
                 model.module.load_state_dict(checkpoint)

            # 옵티마이저 상태가 있다면 로드 (새로운 체크포인트 방식)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("=> loaded optimizer state")

            # 시작할 에폭 번호가 있다면 설정
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # --- 학습 루프 ---
    total_steps = len(train_loader) * args.epochs
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        for i, batch in enumerate(train_loader):
            current_step = epoch * len(train_loader) + i
            images = batch['pixel_values'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            
            # [변경] Mixup 적용 (배치 단위)
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            # 학습률 스케줄링 적용
            new_lr = get_warmup_cosine_decay_lr(current_step, args.base_lr, total_steps, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            optimizer.zero_grad()
            outputs = model(images)
            # 변형된 레이블(소프트 레이블)로 손실 계산
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if global_rank == 0 and (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], LR: {new_lr:.6f}, Loss: {loss.item():.4f}")
                if writer:
                    writer.add_scalar('training_loss', loss.item(), current_step)
                    writer.add_scalar('learning_rate', new_lr, current_step)
        
        # --- 검증 ---
        model.eval()
        correct = 0 
        total = 0
        val_loss_sum = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(local_rank, non_blocking=True)
                labels = batch['label'].to(local_rank, non_blocking=True)
                # 검증 시에는 Mixup을 적용하지 않습니다 (Hard Label 사용).
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # [변경] 검증 결과 동기화 (Robust averaging)
        # 1. 정확도 동기화 (전체 샘플 수와 정답 수 합산)
        total_tensor = torch.tensor(total).to(local_rank)
        correct_tensor = torch.tensor(correct).to(local_rank)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
        # 2. 손실 동기화 (각 rank의 평균 손실을 계산 후, 전역 평균 계산)
        # 이는 배치 수가 rank마다 다를 수 있는 경우에 더 정확합니다.
        local_avg_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_loss_tensor = torch.tensor(local_avg_loss).to(local_rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        
        # 전역 평균 손실 및 정확도
        avg_val_loss = avg_loss_tensor.item() / world_size
        accuracy = 100 * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0

        if global_rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}")
            if writer:
                writer.add_scalar('validation_accuracy', accuracy, epoch)
                writer.add_scalar('validation_loss', avg_val_loss, epoch)

            if (epoch + 1) % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f"vit_checkpoint_epoch_{epoch+1}.pth")
                print(f"Saving checkpoint to {save_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
    if writer:
        writer.close()
    cleanup()

# --- 메인 실행 함수 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Transformer on ImageNet with DDP, Shampoo, and Algoperf Augmentations')
    # 기본 인자
    parser.add_argument('--data-path', type=str, required=True, help='Path to cache Hugging Face datasets')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=300, help='Logging frequency')
    parser.add_argument('--save-interval', type=int, default=15, help='Checkpoint saving frequency')

    # 학습률 관련 인자
    parser.add_argument('--base-lr', type=float, default=0.0013, help='Base learning rate')
    parser.add_argument('--warmup-steps', type=int, default=15000, help='Number of warmup steps')

    # [추가] 데이터 증강 관련 인자 (Algoperf 기본값 설정)
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha (default: 0.2). Set 0 to disable.')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # [추가] 옵티마이저 하이퍼파라미터 인자 (튜닝을 위해 추가)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.1)')
    parser.add_argument('--beta1', type=float, default=0.95, help='Beta1/Momentum (default: 0.9)')
    
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # torchrun을 통해 실행 (LOCAL_RANK, WORLD_SIZE는 환경 변수로 설정됨)
    train(args)
