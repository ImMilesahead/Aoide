import numpy as np
import matplotlib.pyplot as plt

# Character level RNN
# Takes in a string of text as input and predicts the next character
# Uses a one hot encoding on the characters as input

data = 'Hello, World!'#open('input.txt', 'r').read() #'Hello, World!'

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


shape = (vocab_size, 100, vocab_size)

Wxh = 0.01 * np.random.randn(shape[1], shape[0]) * 2 / np.sqrt(shape[0])
Whh = 0.01 * np.random.randn(shape[1], shape[1]) * 2 / np.sqrt(shape[1])
Why = 0.01 * np.random.randn(shape[2], shape[1]) * 2 / np.sqrt(shape[1])

Bh = np.zeros((shape[1], 1))
By = np.zeros((shape[2], 1))

# Inputs is an array of indicies from char_to_ix[data]
def forward(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # Forward pass
    for t in range(len(inputs)):
        # Create one hot vector
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh( np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + Bh )
        ys[t] = np.dot(Why, hs[t]) + By
        ps[t] = np.exp(ys[t]) / np.sum( np.exp( ys[t] ) )
        loss += -np.log( ps[t][targets[t], 0] )
    # Backward pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dBh, dBy = np.zeros_like(Bh), np.zeros_like(By)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dBy += dy

        dh = np.dot(Why.T, dy) + dhnext
        dhRaw = (1 - hs[t]) * hs[t] * dh
        dBh += dhRaw

        dWxh += np.dot(dhRaw, xs[t].T)
        dWhh += np.dot(dhRaw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhRaw)
    for dparam in [dWxh, dWhh, dWhy, dBh, dBy]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dBh, dBy, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + Bh)
        y = np.dot(Why, h) + By
        p = np.exp(y) / np.sum(np.exp(y))
        xi = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[xi] = 1
        ixes.append(xi)
    return ixes
p = 0
n = 0
seq_length = 10
learning_rate = 1e-1
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(Bh), np.zeros_like(By) # memory variables for Adagrad
l = []
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
    if p + seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((shape[1], 1)) # reset RNN memory
        p = 0   # Start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # Sample every now and then
    if n % 10000 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n ----' % (txt, ))

    # Forward seq-length characters through the net and get grad
    loss, dWxh, dWhh, dWhy, dBh, dBy, hprev = forward(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    l.append(smooth_loss)
    if n % 10000 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, Bh, By],
                                [dWxh, dWhh, dWhy, dBh, dBy],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter
    if n % 10000 == 0:
        plt.plot(l)
        plt.show()