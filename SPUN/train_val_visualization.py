import argparse
import os
import torch
import torchvision.transforms as transforms
import yaml
from src.datasets.build import make_dataloader
from config import cfg
from dataset import SPEdataset
from src.loss.build import get_loss
from evaluation import run_evaluation
from src.net.build  import get_model
import matplotlib as plt

def run_evaluation_visualization(model, dataset, loss_function, device, floating_point_type):
    total_loss = 0.0
    num_samples = 0
    model.eval() 
    val_load_iter = iter(dataset)
    eval_fraction = 0.1
    for i in range(int(len(dataset)*eval_fraction)):
        data = val_load_iter.next()
        if floating_point_type == "double":
            target_varr = data["quater"].double().to(device)
            target_vart = data["pos"].double().to(device)
            input_var = data["image"].double().to(device)
        else:
            target_varr = data["quater"].float().to(device)
            target_vart = data["pos"].float().to(device)
            input_var = data["image"].float().to(device)

        if torch.sum(torch.isnan(target_varr)) > 0:
            continue
            # compute output
        output = model(input_var)
        if loss_function.__class__.__name__ == "Nonprobability":
            # norm over the last dimension (ie. orientations)
            print('soon available')
        if loss_function.__class__.__name__ == "BGMixtureLoss":
            loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "BGLloss":
            loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "Separation_distribution":
             loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "S3R3loss":
             loss, log_likelihood = loss_function(target_varr,  target_vart, output)

        if floating_point_type == "double":
            loss = loss.double() / data["image"].shape[0]
        else:
            loss = loss.float() / data["image"].shape[0]
        # measure accuracy and record loss
        total_loss += loss.item()
        num_samples += 1
    # 计算平均 loss
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss
    
def main():
    """Loads arguments and starts testing."""
    device = torch.device(
         'cuda' if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model = get_model(cfg)
    model.to(device)
    print("Model name: {}".format(cfg.model_name))

    

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
        train_dataset = SPEdataset(root, "train", transform)
        val_dataset = SPEdataset(root, "val", transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
        print("dataset is urso")
    elif cfg.data_style == 'dragon':
            use_memory_pinning = True
            root = "./UE4/dragon_hard"
            train_dataset = SPEdataset(root, "train", transform)
            val_dataset = SPEdataset(root, "val", transform)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
            validationloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=use_memory_pinning)
            print("dataset is dragon_hard")
    elif cfg.data_style == 'speed':
            trainloader, validationloader = make_dataloader(cfg, is_train=True, is_source=True)
            print("dataset is speedplus")
    elif cfg.data_style == 'speed_noplus':
            trainloader, validationloader = make_dataloader(cfg, is_train=True, is_source=True)
            print("dataset is speed")

    # Get loss function
    loss_function = get_loss(cfg)

  
    floating_point_type = "float"
    
    train_losses = []
    val_losses = []
    for epoch in range(cfg.max_epochs):
        model_path=os.path.join('models_ckp', cfg.data_style, cfg.loss_name, cfg.model_name, cfg.lr,
                                  'checkpoint_{}_{}.tar'.format(
                                      cfg.model_name, str(epoch)))
    
        print('model_path:', model_path)
    
        if os.path.isfile(model_path):
            print("Loading model {}".format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["state_dict"])
        
        else:
            assert "model not found"
        
        train_loss = run_evaluation_visualization(model, trainloader, loss_function, device, floating_point_type)
        train_losses.append(train_loss)
        val_loss = run_evaluation_visualization(model, validationloader, loss_function, device, floating_point_type)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        # 绘制 loss 曲线
    epochs = range(cfg.max_epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.show()
        

    
    

if __name__ == '__main__':
    main()
