import argparse
import torch
from models import *
from utils import Mydataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

    
def main(args, model, criterion, opt):
    train_data= Mydataset(MNIST(root='./lib', train=True, download=True))
    val_data = Mydataset(MNIST(root='./lib', train=False, download=True))
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.n_w
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.n_w
    )
    min_val_loss = 1.
    patience = 0
    for epoch in range(args.epochs):
        train_loss = 0.
        train_acc = 0.
        model.train()
        for idx, (train_x, train_y) in enumerate(tqdm(train_loader)):
            train_x, train_y = train_x.cuda(), train_y.cuda()
            opt.zero_grad()

            y_, z = model(train_x)
            loss = criterion(y_, train_y)

            train_acc += torch.sum(z == train_y).cpu().numpy()
            train_loss += loss
            loss.backward()
            opt.step()
        print(f"epoch:{epoch}_loss:{round(train_loss.cpu().detach().numpy() / (idx+1), args.precision)}_acc:{round(train_acc / (idx+1) / args.bs, args.precision)}")
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_acc = 0.
            for idx, (val_x, val_y) in enumerate(val_loader):
                val_x, val_y = val_x.cuda(), val_y.cuda()
                y_, z = model(val_x)
                loss = criterion(y_, val_y)
                val_loss += loss
                val_acc += torch.sum(z == val_y).cpu().numpy()
        val_loss = val_loss.cpu().detach().numpy() / (idx+1)
        print(f"validation_loss:{round(val_loss, args.precision)}_acc:{round(val_acc / (idx+1) / args.bs, args.precision)}")
        print(f"patience:{patience}, min_loss:{min_val_loss}, le:{min_val_loss - val_loss>= args.patience[1]}")
        print('-.'*40)
        if min_val_loss - val_loss >= args.patience[1]:
            min_val_loss = val_loss
            patience = 0
            params =  {'net':model.state_dict()}
            torch.save(params, args.paraFile)
            # params =  {}
            # for name, parameter in model.named_parameters():
            #     params[name] = parameter.detach().cpu().numpy()
            # np.savez(args.paraFile, **params)
            del params
        else:
            patience += 1
            if patience == args.patience[0]:
                print(f"Model's training has finished at epoch{epoch}!")
                print("="*40)
                break
                

def export_onnx(args, model):
    params = torch.load(args.paraFile)
    model.load_state_dict(params['net'])
    torch.onnx.export(
                model,
                torch.randn(1, 1, args.img_sz, args.img_sz, device='cuda'),
                args.onnxFile,
                input_names = ['x'],
                output_names = ['y', 'z'],
                do_constant_folding=True,
                verbose=True,
                keep_initializers_as_inputs=True,
                opset_version=12,
                dynamic_axes={
                    'x':{0: 'inBatchSize'},
                    'z':{0: 'outBatchSize'}
                })
    print('onnx file has exported!')

def config():
    parser = argparse.ArgumentParser(description="test tensorRT")
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--w_d', default=1e-4)
    parser.add_argument('--bs', default=256)
    parser.add_argument("--epochs", default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_w', default=4)
    parser.add_argument('--patience', default=(10, 1e-4))
    parser.add_argument('--img_sz', default=28)
    # parser.add_argument('--paraFile', default='./para.npz')
    parser.add_argument('--paraFile', default='./para.pth')
    parser.add_argument('--onnxFile', default='./model.onnx')
    parser.add_argument('--precision', default=4)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = config()
    model = Net().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        params=params,
        lr=args.lr,
        weight_decay=args.w_d
    )
    main(args, model, criterion, opt)
    export_onnx(args, model)