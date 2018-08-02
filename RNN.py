import numpy as np
import matplotlib.pyplot as plt


def get_input_data(x):
    global vocab_size
    global conversion
    X = np.zeros((vocab_size, 1))
    X[conversion[x]] = 1
    return X

def get_output_data(targets):
    # targets should be a string, actually same string as inputs
    Y = [get_input_data(targets[x]) for x in range(1, len(targets)) ]
    return Y

def forward_pass(P, i, ph):
    h = P['Wxh'].dot(i) + P['bh'] + P['Whh'].dot(ph)
    y = P['Why'].dot(h)
    t = np.exp(y)
    b = np.sum(t)
    p = t / b
    print(p)
    print(y)
    return h, y, p

def forward(inputs, P):
    x, h, y, p = {}, {}, {}, {}
    h[-1] = np.zeros_like(P['bh'])
    for i in range(len(inputs)):
        # inputs[i] => char
        x[i] = get_input_data(inputs[i])
        h[i], y[i], p[i] = forward_pass(P, x[i], h[i-1])
    return x, h, y, p

def train(inputs, P, targets = None):
    x, h, _, p = forward(inputs, P)
    grads = {
        'dWxh': np.zeros_like(P['Wxh']),
        'dWhh': np.zeros_like(P['Whh']),
        'dWhy': np.zeros_like(P['Why']),

        'dbh': np.zeros_like(P['bh']),
        'dby': np.zeros_like(P['by'])
    }
    if targets == None:
        # Assume prediction
        Y = get_output_data(inputs)
        dhnext = np.zeros_like(h[0])
        loss = 0
        for i in reversed(range(len(inputs))):
            dy = np.copy(p[i])
            dy -= Y[i-1]
            grads['dWhy'] += np.dot(dy, h[i].T)
            grads['dby'] += dy

            dh = np.dot(P['Why'].T, dy) + dhnext
            dhraw = (1 - h[i]) * h[i] * dh
            grads['dbh'] += dhraw

            grads['dWxh'] += np.dot(dhraw, x[i].T)
            grads['dWhh'] += np.dot(dhraw, h[i-1].T)
            dhnext = np.dot(P['Whh'].T, dhraw)
            loss += -np.log( p[i][np.argmax(Y[i-1]), 0] + 1e-8)

        for key, value in grads.items():
            np.clip(value, -5, 5, out=value) 
        return grads, loss
    else:
        # use targets
        shape = targets.shape
        if len(shape) == 1:
            # Assume many to one
            pass
        else:
            # Assume one target for each inputs
            pass

def sample(inputs, P, n=25, s=1):
    x, h, y, p = {}, {}, {}, {}
    h[-1] = np.zeros_like(P['bh'])
    ixes = []
    for i in range(n+s):
        if i < s:
            x[i] = get_input_data(inputs[i])
        else:
            x[i] = np.zeros_like(x[i-1])
            try:
                xi = np.random.choice(range(x[i].shape[0]), p=p[i-1].ravel())
            except:
                print(p[i-1])
            x[i][xi] = 1
            ixes.append(xi)
        h[i], y[i], p[i] = forward_pass(P, x[i], h[i-1])
    return ixes

def update_parameters(P, G, M, lr):
    for key, value in P.items():
        M[key] = G['d' + key] * G['d' + key]
        P[key] -= learning_rate * G['d' + key] / np.sqrt(M[key] + 1e-8)
    return P, M
if __name__ == '__main__':
    data = 'Hello, World!'
    chars = list(set(data))
    vocab_size = len(chars)
    conversion = {ch:i for i, ch in enumerate(chars) }
    conversion_back = {i:ch for i,ch in enumerate(chars) }
    input_size = vocab_size
    hidden_size = 100
    output_size = vocab_size

    shape = (input_size, hidden_size, output_size)
    
    learning_rate = 1e-1

    sample_size = 200
    s = 1

    P = {
        'Wxh': np.random.randn(hidden_size, input_size) * 0.01,
        'Whh': np.random.randn(hidden_size, hidden_size) * 0.01,
        'Why': np.random.randn(output_size, hidden_size) * 0.01,

        'bh': np.zeros((hidden_size, 1)),
        'by': np.zeros((output_size, 1))
    }
    M = {
        'Wxh': np.zeros_like(P['Wxh']),
        'Whh': np.zeros_like(P['Whh']),
        'Why': np.zeros_like(P['Why']),

        'bh': np.zeros_like(P['bh']),
        'by': np.zeros_like(P['by'])
    }
    sl = -np.log(1.0/vocab_size)*len(data)
    l = []
    for j in range(100000):
        grads, loss = train(data, P)
        sl = sl * 0.999 + loss * 0.001
        l.append(sl)
        P, M = update_parameters(P, grads, M, learning_rate)
        if (j) % 10000 == 0:
            r = sample(data, P, sample_size, s)
            print('Iteration: %d' % j)
            print(''.join(conversion_back[i] for i in r))
            plt.plot(l)
            plt.show()
