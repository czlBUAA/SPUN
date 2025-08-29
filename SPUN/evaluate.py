import argparse
import os
import torch
import torchvision.transforms as transforms
import yaml
from src.datasets.build import make_dataloader,make_dataloader2, make_dataloader1
from config import cfg
from dataset import SPEdataset
from src.loss.build import get_loss
from evaluation_simple import run_evaluation
from src.net.build  import get_model

DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/upna_train.yaml"


def main():
    """Loads arguments and starts testing."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device(
         'cuda' if torch.cuda.is_available() else "cpu")

    model = get_model(cfg)
    model.to(device)
    print("Model name: {}".format(cfg.model_name))
    #folder_path = os.path.join('models_ckp', cfg.data_style, cfg.loss_name, cfg.model_name, str(cfg.lr), str(cfg.L2reg), str(cfg.rot_image), str(cfg.max_epochs),'lightbox')
    folder_path = '/media/computer/study/CZL/Bingham-Guass/models_ckp/speed/sdn/resnet50/0.001/0.0/False/100/sunlamp'
    model_path=os.path.join(folder_path,
                                  'checkpoint_{}_{}.tar'.format(
                                      cfg.model_name, 99))
    
    print('model_path:', model_path)
    
    if os.path.isfile(model_path):
        print("Loading model {}".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        
    else:
        assert "model not found"
        checkpoint = torch.load(model_path)
        print("model not found!!!!!!!!!")
        

    # Get data loader
    root = "./UE4/SPE"
    transform = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    if cfg.data_style == 'urso':
            use_memory_pinning = True
            test_dataset = SPEdataset(root, "test", transform)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
    elif cfg.data_style == 'symmertic':
        use_memory_pinning = True
        root = "./UE4/symmertic"
        test_dataset = SPEdataset(root, "test", transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
    elif cfg.data_style == 'symmertic_pose':
        use_memory_pinning = True
        root = "./UE4/symmertic_pose"
        test_dataset = SPEdataset(root, "test", transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is", cfg.data_style)
    elif cfg.data_style == 'dragon':
        use_memory_pinning = True
        root = "./UE4/dragon_hard"
        test_dataset = SPEdataset(root, "test", transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
    else:
        if cfg.train_domain == 'synthetic':
            testloader = make_dataloader(cfg, is_train=False, is_source=False, is_test=True) #synthetic validation images
            print("dataset is speedplus")
        else:
            testloader  = make_dataloader2(cfg, is_train=True, is_source=True, is_test=True)
            print("dataset is", cfg.data_style,"plus")
    # Get loss function
    loss_function = get_loss(cfg)

  
    floating_point_type = "float"
   

    run_evaluation(
        model, testloader, loss_function, 
        device, floating_point_type
    )
    

if __name__ == '__main__':
    main()
