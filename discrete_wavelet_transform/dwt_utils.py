from pathlib import Path
import numpy as np
from PIL import Image

def load_grayscale(filename):
    return Image.open(filename).convert("L")

def haar_dwt(image):
    arr=np.asarray(image,dtype=np.float32)
    arr=arr[:arr.shape[0]//2*2,:arr.shape[1]//2*2]
    a,b=arr[0::2,0::2],arr[0::2,1::2]
    c,d=arr[1::2,0::2],arr[1::2,1::2]
    return (a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2

def normalize_band(band):
    lo,hi=band.min(),band.max()
    if hi==lo: return np.zeros_like(band,dtype=np.uint8)
    return np.clip((band-lo)*255/(hi-lo),0,255).astype(np.uint8)

def save_subbands(LL,LH,HL,HH,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name,band in [("LL",LL),("LH",LH),("HL",HL),("HH",HH)]:
        Image.fromarray(normalize_band(band),"L").save(output/f"{name}.png")

def make_montage(LL,LH,HL,HH,filename):
    bands=[normalize_band(v) for v in (LL,LH,HL,HH)]
    h,w=bands[0].shape
    out=np.zeros((h*2,w*2),dtype=np.uint8)
    out[:h,:w]=bands[0]; out[:h,w:]=bands[1]; out[h:,:w]=bands[2]; out[h:,w:]=bands[3]
    Image.fromarray(out,"L").save(filename)
