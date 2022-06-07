from torch import nn
from torch.nn import CrossEntropyLoss
import configargparse
from transformers import ViTFeatureExtractor, ViTModel

class ViTEmotion(nn.Module):
    def __init__(self, params:configargparse.Namespace):
        
        super(ViTEmotion,self).__init__()
        self.ViTFeatureExtractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.ViTModel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        hidden_size = self.ViTModel.config.hidden_size
        self.num_labels = params.num_labels
        self.Linear = nn.Linear(hidden_size,self.num_labels)
        
    def forward(self, x, labels):
        output = self.ViTFeatureExtractor(images=x, return_tensors="pt")
        output = output.pixel_values.cuda()
        output = self.ViTModel(pixel_values=output)
    
        output = output[0]
        logits = self.Linear(output[:,0,:])
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits
