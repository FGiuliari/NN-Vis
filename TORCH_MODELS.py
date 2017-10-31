#####  http://pytorch.org/


import torch
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


print('carico immagine')
target_shape = (224, 224)

image = Image.open('data/n02488291_1177.jpg')
if image.size != target_shape:
    image = image.resize(target_shape, Image.NEAREST)

# eventuale conversione in numpy
#image = np.array(image)
#image = np.swapaxes(image, 0, 1) # convert to format H*W

print('la trasformo in pytorch')
# i dati sono già normalizzati tra 0 e 1, quindi rimuovo 0.5 per centrare in 0 l'intero dataset
# e moltiplico per 2 (dividendo per 0.5) in modo da scalare il dataset tra -1 e +1
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trasforma l'immagine PIL in immagine PyTorch (tensore)
# che è normalizzata tra -1 e +1
image = transf(image)


print('carico modello vgg16 con batch normalization')
model = models.vgg16_bn(pretrained=True)
model.eval() # congela i gradienti (salva memoria e velocizza le prestazioni)
print(model)


print('controllo accesso cuda')
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

if HAS_CUDA:
    gpu_id = 0
    model.cuda(gpu_id)


'''

x è un tensore di pytorch convertibile in numpy

x.numpy()

'''

print('processing')
image = image if not HAS_CUDA else image.cuda(gpu_id)
image = Variable(image) # wrapper che gestisce i gradienti
image = image.unsqueeze(0) # mettin in posizione 0 (prima) una nuova dimensione
predictions = model(image) # in pytorch!
predictions = predictions.data.cpu() # sposto il tensore dalla gpu alla ram
predictions = predictions.numpy() # trasformo il tensore pytorch in numpy

print(predictions)

idx = np.argmax(predictions)
print(idx)