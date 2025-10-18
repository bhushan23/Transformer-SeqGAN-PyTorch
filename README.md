# Transformer-SeqGAN-PyTorch
Merging Transformer Nets with SeqGAN in PyTorch

Natural Language generation involves generation of Structured language from unstructured data. Till now, the state of the art implementations uses Recurrent Neural Network (RNN) based models. These are slow and take a long time to train. To improve the training time and achieve similar results, we use a self attention based Transformer Network. As Generative Adversarial Networks (GAN) are known to perform well for realistic data generation tasks in general, we combine the training approach as in SeqGAN and Transformer Network for language generation. We implemented the Transformer Network for Language Generation task. We were also able to implement the GAN model with Transformer Network as the Generator and generated results for Obama Speech Sentences

## Reference Sources
Our Work extends existing SeqGAN and Transformer Model.
 
### 1. SeqGAN-PyTorch 
   - [Implementation of SeqGAN in PyTorch, following the implementation in tensorflow](https://github.com/ZiJianZhao/SeqGAN-PyTorch)
   - Paper- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf)

### 2. Transformer Net
   - [Implementation Transformer model in PyTorch](https://github.com/phohenecker/pytorch-transformer)
   - Paper - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

## Source Description
```
├─── flask_app/ : Web Based app to query results <br>
     ├─── app.py       - Web App starting script <br>
     ├─── interface.py - Interface loading all the models and book-keeping metadata <br>
     ├─── test_app.py  - Testing script <br>
├─── checkpoints/ - Directory containing checkpoints and metadata required for flask app <br>
├─── attention-only/ : Transformer based language generation <br>
     ├─── train_obama.py - Script to train attention based model on obama dataset. <br>
                         - Loads dataset, builds vocabulary, trains and tests model <br>
├─── seq_gan/ : SeqGAN based language generation <br>
     ├─── main.py - Script to train and test SeqGAN based model generation <br>
├─── seq_gan_with_attention/ : Transformer based SeqGAN for language generation <br>
     ├─── main.py - Script to train and test Transformer based SeqGAN <br>
     ├─── loss.py - Loss functions <br>
     ├─── generator_attention.py - Transformer based generator <br>
├─── core/ : Core routines shared across the model <br>
     ├─── data_iter.py - Data Iterator for Generator and Discriminator and Testing <br>
     ├─── helper.py    - Helper routines such as loading file, building vocab, padding, generating sentences from ids, storing and loading checkpoints.
```

## New Functionalities added
```
1. Flask app for testing
2. Core utility for data processing
   core/helper.py: Library for pre-processing data, loading samples and generating sentences from ids
3. seq_gan/train_obama.py
    - Updated input and output fed to prepare_data()
4. seq_gan_with_attention/
    - rollout.py
        - get_reward(): Update the reward functin to pad zeros for consisting while computing rewards
    - main.py
        - Update loss to binary cross entropy during Pre-training
        - Update GAN training
            Earlier, samples generated from GAN were fed in to train the generator
            Fixing this to pick up new samples from train data and using generator output samples for computing reward.
    - generator_attention.py
        - Implemented Generator model using Transformer model internally.
 ```
## Training and Testing
### Training Attention-Only
```
# 1. Go to attention-only directory
cd attention-only
# Edit train_obama.py to update following parameters
# 1. Number of sentences to load from file: Control in load_from_big_file() function
#    Also, control train-test split here.
# 2. set NUM_EPOCHS to desired epoch number to train
# Train and test
python train_obama.py

```

### Training and Testing Seq-GAN
```
# 1. Go to Seq-GAN directory
cd seq_gan
# Edit main.py to update following parameters
# 1. File to load data from and number of sentences to load- load_from_big_file() 
# 2. TOTAL_BATCH: Number of epochs for adversarial training
# 3. BATCH_SIZE:  Batch size being used
# 4. ROOT_PATH:   Path to be used for storing checkpoints and metadata
# 5. POSITIVE_FILE, NEGATIVE_FILE, DEBUG_FILE, EVAL_FILE: store real data, generator generated data, debug data and evaluationd data respectively
# 6. Vocab size: 5000 # can ignore as will be updated after load_from_big_file()
# 7. PRE_EPOCH_NUM: Pre-Training epochs
# Train and test
# Train on CPU
python main.py
# or train on GPU
python main.py --cuda <device number>
# Testing 
python main.py --test <--cuda <device number>>
```

### Training and Testing Transformer based Seq-GAN
```
# 1. Go to Seq-GAN directory
cd seq_gan
# Edit main.py to update following parameters
# 1. File to load data from and number of sentences to load- load_from_big_file() 
# 2. TOTAL_BATCH: Number of epochs for adversarial training
# 3. BATCH_SIZE:  Batch size being used
# 4. ROOT_PATH:   Path to be used for storing checkpoints and metadata
# 5. POSITIVE_FILE, NEGATIVE_FILE, DEBUG_FILE, EVAL_FILE: store real data, generator generated data, debug data and evaluationd data respectively
# 6. Vocab size: 5000 # can ignore as will be updated after load_from_big_file()
# 7. PRE_EPOCH_NUM: Pre-Training epochs
# Train and test
# Train on CPU
python main.py
# or train on GPU
python main.py --cuda <device number>
# Testing 
python main.py --test <--cuda <device number>>
```
    
### Testing Through Interface
```
# following needs all the checkpoints in flask_app/checkpoints as mentioned in flask_app/interface.py
cd flask_app
python test_app.py
```

### Testing Through Flask-App
```
# following needs all the checkpoints in flask_app/checkpoints as mentioned in flask_app/interface.py
cd flask_app
python app.py
```

Sample Input
![Sample Input](https://github.com/bhushan23/Transformer-SeqGAN-PyTorch/blob/master/flask_app/input.png)
Sample Output
![Sample Output](https://github.com/bhushan23/Transformer-SeqGAN-PyTorch/blob/master/flask_app/output.png)


## Requirements: 
* PyTorch v0.1.12+
* Python 3.6
* CUDA 7.5+ (For GPU)

## Cite

If you find this repo useful, you can cite this in your research as follows

@misc{transformer_seqgan,
  author = {Bhushan Sonawane, Nishant Borude, Mihir Chakradeo},
  title = {Merging Transformer Nets with SeqGAN in PyTorch},
  year = {2018},
  publisher = {GitHub},
  url = {https://github.com/bhushan23/Transformer-SeqGAN-PyTorch}
}
