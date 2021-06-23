import matplotlib.pyplot as plt
font_titles = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,}
font_labels = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 13,}



import numpy as np
time_scale = np.arange(0.01,0.51,0.01)

mae_hybrid_model_close_training=np.load('mae_error_hybrid_model.npy')

mae_seq2seq_model_close_training=np.load('mae_error_seq2seq_model.npy')

#索引
uy_mae_hybrid_model_close_training=mae_hybrid_model_close_training[:,:,0:1].reshape(-1,50)

uy_mae_seq2seq_model_close_training=mae_seq2seq_model_close_training[:,:,0:1].reshape(-1,50)



r_mae_hybrid_model_close_training=mae_hybrid_model_close_training[:,:,1:2].reshape(-1,50)

r_mae_seq2seq_model_close_training=mae_seq2seq_model_close_training[:,:,1:2].reshape(-1,50)


uy_mean_mae_hybrid_model_close_training=np.mean(uy_mae_hybrid_model_close_training,axis=0)

uy_mean_mae_seq2seq_model_close_training=np.mean(uy_mae_seq2seq_model_close_training,axis=0)


r_mean_mae_hybrid_model_close_training=np.mean(r_mae_hybrid_model_close_training,axis=0)

r_mean_mae_seq2seq_model_close_training=np.mean(r_mae_seq2seq_model_close_training,axis=0)

#画uy
plt.figure(dpi=120,figsize=(15,5))

plt.subplot(121)
plt.xlabel('Predicted Horizens (s)',fontdict=font_labels)
plt.ylabel('$||e_{uy,k}||$ (m/s)',fontdict=font_labels)
plt.grid(True)
plt.title('$T_{pred}=0.5sec\ @100Hz $')
plt.plot(time_scale,uy_mean_mae_seq2seq_model_close_training,label='GRU Encoder-Decoder',lw=1,marker='o',markerfacecolor='none')
plt.plot(time_scale,uy_mean_mae_hybrid_model_close_training,label='Hybrid model',lw=1,marker='s',markerfacecolor='none')
plt.legend(fontsize=10,loc=0,ncol=1)

#画r

plt.subplot(122)
plt.xlabel('Predicted Horizens (s)',fontdict=font_labels)
plt.ylabel('$||e_{r,k}||$ (rad/s)',fontdict=font_labels)
plt.grid(True)
plt.title('$T_{pred}=0.5sec\ @100Hz $')
plt.plot(time_scale,r_mean_mae_seq2seq_model_close_training,label='GRU Encoder-Decoder',lw=1,marker='o',markerfacecolor='none')
plt.plot(time_scale,r_mean_mae_hybrid_model_close_training,label='Hybrid model',lw=1,marker='s',markerfacecolor='none')
plt.legend(fontsize=10,loc=0,ncol=1)
plt.show()