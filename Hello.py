from __future__ import print_function
# import math
from flask import Flask, render_template,request
from forms import InputForm
# import os
import sys
import torch
import io
import time
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
import string
import os
import time
import random
import pickle 
# from matplotlib.pyplot import imshow

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))
gm = os.path.join('static', 'generated_images')
image_path = 'static/generated_images/'
app.config['UPLOAD_FOLDER']= gm
global finished
def getImagefromArray(imgs):
	for i in imgs:
		sample_images.append(Image.fromarray(imgs[i]))
	return sample_images

def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    #print(captions.astype(int), cap_lens.astype(int))
    #captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])
    #print(captions, cap_lens)
    #return captions, cap_lens

    return captions.astype(int), cap_lens.astype(int)

def imglist(fake_imgs,batch_size):
	sample_images = list()
	imgs = []
	for j in range(batch_size):
		for k in range(len(fake_imgs)):
			im = fake_imgs[k][j].data.cpu().numpy()
			im = (im + 1.0) * 127.5
			im = im.astype(np.uint8)
			im = np.transpose(im, (1, 2, 0))
			# im = Image.fromarray(im)
			imgs.append(im)
		path = ''.join(random.choices(string.ascii_uppercase + string.digits+string.ascii_lowercase, k=8))+".jpg"
		image= Image.fromarray(imgs[-1])
		image.save(image_path+path)
		# time.sleep(10)
		sample_images.append(os.path.join(app.config['UPLOAD_FOLDER'],path))
	return sample_images

def generate(captions,copies):
	x = pickle.load(open('data/captions.pickle', 'rb'))
	# print(x)
	ixtoword = x[2]
	wordtoix = x[3]
	del x
	word_len = len(wordtoix)

	text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
	state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
	text_encoder.load_state_dict(state_dict)
	text_encoder.eval()
        
	netG = G_NET()
	state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
	netG.load_state_dict(state_dict)
	# netG.eval()
	# seed = 100
	# random.seed(seed)
	# np.random.seed(seed)
	# torch.manual_seed(seed)
    # load word vector
	captions, cap_lens  = vectorize_caption(wordtoix, captions, copies)
	n_words = len(wordtoix)

    # only one to generate
	batch_size = captions.shape[0]
	nz = cfg.GAN.Z_DIM
	captions = Variable(torch.from_numpy(captions), volatile=True)
	cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
	noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
    

    #######################################################
    # (1) Extract text embeddings
    #######################################################
	hidden = text_encoder.init_hidden(batch_size)
	words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
	mask = (captions == 0)
        

    #######################################################
    # (2) Generate fake images
    #######################################################
	noise.data.normal_(0, 1)
	fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # G attention
	cap_lens_np = cap_lens.cpu().data.numpy()

 #    # storing to blob storage
	# container_name = "images"
	# full_path = "https://attgan.blob.core.windows.net/images/%s"
	# prefix = datetime.now().strftime('%Y/%B/%d/%H_%M_%S_%f')
	imgs = []
    # only look at first one
    #j = 0
	return imglist(fake_imgs,batch_size)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/text',methods = ['POST', 'GET'])
def start_generation():
	sample_images= generate(request.form['text'],9)
	end_time= time.localtime(time.time() )
	# time_taken = abs(m1-m2)*60+abs(s1-s2)
	return render_template("hii.html",results=sample_images)


if __name__ == '__main__':
	app.run(host='127.0.0.1',port=5000,debug=True)