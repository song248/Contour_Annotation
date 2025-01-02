import torch
from model.models import UNet
from torchvision.transforms import ToTensor, Compose, Normalize

class UNetSingleInference(object):
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)

        self.model.eval()
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=0.5,std=0.5)
        ])
        self.fn_class = lambda x: 1.0 * (x > 0.5)
        self.fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    def load_model(self, model_path):
        net = UNet()
        # model_dict = torch.load(model_path)
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        net.load_state_dict(model_dict['net'])
        net.to(self.device)
        return net

    def inference(self,image):
        #image to tensor
        image = self.transform(image)#.unsqueeze(0)
        image = image[None, ::]
        image = image.to(self.device)
        output = self.model(image)
        output = self.fn_tonumpy(self.fn_class(output))

        return output