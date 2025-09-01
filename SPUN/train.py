import os
import sys
import torch
import os.path as osp
import logging
import torch.optim as optim
from torchvision.models import resnet50
import torchvision.transforms as transforms
from dataset import SPEdataset
import net
from src.net.build  import get_model, get_optimizer
from src.loss.build import get_loss
from trainer import Trainer
from src.datasets.build import make_dataloader, make_dataloader1
from config import cfg
from src.styleaug.styleAugmentor import StyleAugmentor
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)
def load_latest_checkpoint(model, folder_path, model_name='resnet50'):
    # 判断文件夹是否存在
    if not os.path.exists(folder_path):
        print("文件夹不存在，从头开始训练。")
        return 0

    # 获取文件夹中所有.tar文件
    tar_files = [f for f in os.listdir(folder_path) if f.endswith('.tar')]
    
    if not tar_files:
        print("文件夹中没有.tar文件，从头开始训练。")
        return 0

    # 提取文件名中的epoch值，假设文件命名格式为: ckp_resnet50_{epoch}.tar
    epochs = []
    for filename in tar_files:
        try:
            epoch_str = filename.split('_')[-1].split('.')[0]
            epoch_val = int(epoch_str)
            epochs.append(epoch_val)
        except ValueError:
            continue

    if not epochs:
        print("未找到合法的checkpoint文件，从头开始训练。")
        return 0

    # 找到最大的epoch值
    max_epoch = max(epochs)
    checkpoint_path = os.path.join(folder_path, f'checkpoint_{model_name}_{max_epoch}.tar')
    print("load ckp file:", checkpoint_path)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    # 返回下一个epoch作为训练起始编号
    return max_epoch+1
def load_checkpoint(checkpoint_file, model, optimizer, device):
    load_dict = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(load_dict['state_dict'], strict=True)
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
def save_checkpoint1(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    logger.info('Checkpoint saved to {}'.format(os.path.join(output_dir, filename)))

    if is_best and 'state_dict' in states:
        torch.save(
            states['state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )
        logger.info('Best model saved to {}'.format(os.path.join(output_dir, 'model_best.pth.tar')))
#torch.manual_seed(0)
torch.manual_seed(0)
root = "./UE4/SPE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device(
         'cuda' if torch.cuda.is_available() else "cpu")

print("Using device: {}".format(device))
transform = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
if cfg.data_style == 'urso':
        use_memory_pinning = True
        train_dataset = SPEdataset(root, "train", transform)
        val_dataset = SPEdataset(root, "val", transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
elif cfg.data_style == 'symmertic_pose':
        use_memory_pinning = True
        root = "./UE4/symmertic_pose"
        train_dataset = SPEdataset(root, "train", transform)
        val_dataset = SPEdataset(root, "val", transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
elif cfg.data_style == 'symmertic':
        use_memory_pinning = True
        root = "./UE4/symmertic"
        train_dataset = SPEdataset(root, "train", transform)
        val_dataset = SPEdataset(root, "val", transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
elif cfg.data_style == 'dragon':
        use_memory_pinning = True
        root = "./UE4/dragon_hard"
        train_dataset = SPEdataset(root, "train", transform)
        val_dataset = SPEdataset(root, "val", transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
elif cfg.data_style == 'speed':
        if cfg.train_domain == 'synthetic':
            trainloader = make_dataloader(cfg, is_train=True, is_source=True)
            validationloader = make_dataloader(cfg, is_train=False, is_source=False)
            print("dataset is", cfg.data_style,"plus")
        else:
            trainloader, validationloader = make_dataloader1(cfg, is_train=True, is_source=True)
            print("dataset is", cfg.train_domain)
elif cfg.data_style == 'speed_noplus':
        trainloader, validationloader = make_dataloader(cfg, is_train=True, is_source=True)
        print("dataset is speed")
        
            
# This should not be necessary but it surprisingly is. In the presence of a
# GPU, PyTorch tries to allocate GPU memory when pin_memory is set to true
# in the data loader. This happens even if training is to happen on CPU and
# all objects are on CPU.


# Pose estimation CNN
model = get_model(cfg)
print("network is", cfg.model_name)
model.to(device)
# Style Augmentor
styleAugmentor = None
if cfg.randomize_texture:
    styleAugmentor = StyleAugmentor(cfg.texture_alpha, device)
    logger.info('Texture randomization enabled with alpha = {}'.format(cfg.texture_alpha))
    logger.info('   - Randomization ratio: {:.2f}'.format(cfg.texture_ratio))
loss_function = get_loss(cfg)
print("loss is", cfg.loss_name)




optimizer = get_optimizer(cfg, model)
# Load checkpoint
checkpoint_file = osp.join(cfg.savedir, 'checkpoint.pth.tar')
if cfg.auto_resume and osp.exists(checkpoint_file):
    last_epoch, best_speed = load_checkpoint(checkpoint_file, model, optimizer, device)
    begin_epoch = last_epoch
    best_perf   = begin_epoch
else:
    begin_epoch = 0
    # best_perf   = 1e10
    best_perf   = begin_epoch
#miXed-precision training
scaler = None
if cfg.fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info('Mixed-precision training enabled')
    # LR scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay_alpha)

trainer = Trainer(device, 'float')
print("rot_image", cfg.rot_image)
#filename_train=os.path.join('log', cfg.data_style, cfg.loss_name, cfg.model_name, str(cfg.lr), str(cfg.L2reg), str(cfg.rot_image),str(cfg.max_epochs), str(cfg.mix_count), 'training') #'rot_camera_img',L2reg0.001, 
#filename_val=os.path.join('log', cfg.data_style, cfg.loss_name, cfg.model_name, str(cfg.lr), str(cfg.L2reg), str(cfg.rot_image), str(cfg.max_epochs),str(cfg.mix_count),'validation')
filename_train=os.path.join('log','5fold_sunlamp', '1th_fold', 'training') #'rot_camera_img',L2reg0.001, 
filename_val=os.path.join('log', '5fold_sunlamp', '1th_fold','validation')
# filename_train=os.path.join('log', 'mutimodal', 'lightbox', 'training') #'rot_camera_img',L2reg0.001, 
# filename_val=os.path.join('log', 'mutimodal', 'lightbox','validation')
writer_train = SummaryWriter(filename_train) 
writer_val = SummaryWriter(filename_val)


#folder_path = os.path.join('models_ckp', cfg.data_style, cfg.loss_name, cfg.model_name, str(cfg.lr), str(cfg.L2reg), str(cfg.rot_image), str(cfg.max_epochs),str(cfg.mix_count))
folder_path = os.path.join('models_ckp', '5fold_sunlamp', '1th_fold')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
start_epoch =load_latest_checkpoint(model, folder_path, model_name=cfg.model_name)
print("start training epoch:", start_epoch)
# Main loop
perf    = 1e10
is_best = False
for epoch in range(start_epoch, cfg.max_epochs+start_epoch):
    trainer.train_epoch(trainloader,model, loss_function, optimizer,epoch, 
    writer_train, writer_val, validationloader, styleAugmentor=styleAugmentor)
    lr_scheduler.step()
    # save_checkpoint(
    #         {'epoch': epoch + 1, 'state_dict': model.state_dict()},
    #         filename=os.path.join(folder_path,
    #                               'checkpoint_{}_{}.tar'.format(
    #                                   cfg.model_name, epoch))
    #     )
    perf = epoch+1
    if perf > best_perf:
        best_perf = perf
        is_best = True
    else:
        is_best = False

    # Save
    save_checkpoint1({
        'epoch': epoch + 1,
        'model': cfg.model_name,
        'state_dict': model.state_dict(),
        'best_score': best_perf,
        'optimizer': optimizer.state_dict(),
    }, is_best, folder_path)
print('Finished training')
