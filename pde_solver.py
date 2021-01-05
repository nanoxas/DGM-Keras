import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_siren import SinusodialRepresentationDense
import skimage
import tensorflow_probability as tfp
import time
from matplotlib import animation


def get_grads(x):
    return K.gradients(x[0], x[1])[0]


def get_zeros(x):
    return tf.zeros_like(x)


def heat_pde_loss(y_true, y_pred):
    utt = y_pred[..., 0]
    uxx = y_pred[..., 1:3]
    uxx = K.sum(uxx, axis=-1)
    ut_t0 = y_pred[..., 3]
    eq_res = K.square(utt - uxx)

    return eq_res + K.square(ut_t0)


def ic_loss(y_true, y_pred):
    return K.square(y_true - y_pred)


def bc_loss(y_true, y_pred):
    return K.square(y_true - y_pred)


def dirac_delta(x):
    m_rect_x = np.where(np.abs(x[..., 0]) <= 0.1, np.ones_like(x[..., 0]), np.zeros_like(x[..., 0]))
    m_rect_y = np.where(np.abs(x[..., 1]) <= 0.1, np.ones_like(x[..., 1]), np.zeros_like(x[..., 1]))

    return m_rect_x * m_rect_y


def get_c(x):
    m_rect_x = np.where(np.abs(x[..., 0]) <= 0.5, np.ones_like(x[..., 0]), np.zeros_like(x[..., 0]))
    m_rect_y = np.where(np.abs(x[..., 1]) <= 0.5, np.ones_like(x[..., 1]), np.zeros_like(x[..., 1]))
    c_inner = m_rect_x * m_rect_y
    c_inner[c_inner == 0] = 1
    return c_inner


def get_model(x_shape):
    input_t = Input((1,))
    input_x = Input(x_shape)
    f_x = Flatten()(input_x)
    c_inpt = Concatenate()([input_t, f_x])
    x = Dense(256, activation='tanh')(c_inpt)
    x = SinusodialRepresentationDense(256, activation='sine', w0=1.0)(x)
    x = SinusodialRepresentationDense(256, activation='sine', w0=1.0)(x)
    x = SinusodialRepresentationDense(256, activation='sine', w0=1.0)(x)
    sol = Dense(1)(x)
    model = Model([input_t, input_x], sol)
    model.summary()
    return model


def slice_tensor_x(x):
    return x[..., 0:1]


def slice_tensor_y(x):
    return x[..., 1:]


def pde_model(x_shape):
    input_t = Input((1,))
    input_x = Input(x_shape)
    s_x = Lambda(slice_tensor_x)(input_x)
    s_y = Lambda(slice_tensor_y)(input_x)
    i_x = Concatenate()([s_x, s_y])
    input_bc = Input(x_shape)
    t_init = Lambda(get_zeros)(input_t)
    heat_model = get_model(x_shape)

    init_sol = heat_model([t_init, i_x])
    bc_sol = heat_model([input_t, input_bc])
    gen_sol = heat_model([input_t, i_x])

    ut = Lambda(get_grads)([gen_sol, input_t])
    ut_t0 = Lambda(get_grads)([init_sol, t_init])
    utt = Lambda(get_grads)([ut, input_t])

    ux = Lambda(get_grads)([gen_sol, s_x])
    uxx = Lambda(get_grads)([ux, s_x])
    uy = Lambda(get_grads)([gen_sol, s_y])
    uyy = Lambda(get_grads)([uy, s_y])

    out_grads = Concatenate()([utt, uxx, uyy, ut_t0])

    model = Model([input_t, input_x, input_bc], [bc_sol, init_sol, out_grads])
    model.summary()
    return model, heat_model


def train_model():
    steps = int(5e4)
    batch_size = 1024
    model, inf_model = pde_model((2,))
    decay_steps = 10000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(1e-3, decay_steps)
    opt = Adam(1e-3)
    model.compile(loss=[bc_loss, ic_loss, heat_pde_loss], loss_weights=[1, 1, 1], optimizer=opt)
    avg_error = 0
    for step in range(steps):
        t_train = np.random.uniform(0, 1, (batch_size, 1))
        x_train = np.random.uniform(-1, 1, (batch_size, 2))
        c_train = np.expand_dims(get_c(x_train), axis=-1)
        ic_gt = dirac_delta(x_train)
        bc_x = np.random.choice([-1, 1], size=(batch_size // 2, 1))
        bc_y = np.random.choice([-1, 1], size=(batch_size // 2, 1))
        bc_rx = np.random.uniform(-1, 1, (batch_size // 2, 1))
        bc_ry = np.random.uniform(-1, 1, (batch_size // 2, 1))
        bc_xx = np.concatenate((bc_x, bc_rx), axis=-1)
        bc_yy = np.concatenate((bc_ry, bc_y), axis=-1)
        bc_x = np.concatenate((bc_xx, bc_yy), axis=0)

        loss_outer = model.train_on_batch([t_train, x_train, bc_x],
                                          [np.zeros_like(ic_gt), ic_gt, np.concatenate((c_train, x_train), axis=-1)])
        if step % 100 == 0:
            u = (1 / np.sqrt(4 * np.pi * t_train)) * np.exp((-x_train ** 2) / (4 * t_train))
            error = np.sum(np.square(u - inf_model.predict([t_train, x_train])[0][..., 0])) / np.sum(np.square(u))
            avg_error += error
            print('step: ' + str(step) + 'loss: ' + str(loss_outer))

    return inf_model


#inf_model = train_model()
#inf_model.save_weights('./wave_pde.h5')
inf_model = get_model((2,))
inf_model.load_weights('./wave_pde.h5')

k = 1
x = np.outer(np.linspace(-1, 1, 50), np.ones(50))
y = np.outer(np.linspace(-1, 1, 50), np.ones(50)).T
x_inp = np.linspace(-1, 1, 50)
y_inp = np.linspace(-1, 1, 50)
t_inp = np.linspace(0, 1, 50)
u_pred = np.zeros((t_inp.shape[0], y_inp.shape[0], y_inp.shape[0]))
for i in range(t_inp.shape[0]):
    print(i)
    t_x = np.ones_like(y_inp) * t_inp[i]
    for j in range(y_inp.shape[0]):
        y_x = np.expand_dims(np.ones_like(y_inp) * y_inp[j], axis=-1)
        xy = np.concatenate((np.expand_dims(x_inp, axis=-1), y_x), axis=-1)
        u_pred[i, :, j] = inf_model.predict([t_x, xy])[..., 0]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Surface plot')
implot = ax.plot_surface(x, y, u_pred[0], cmap='viridis', edgecolor='none')


def update(i, ax, fig):
    ax.cla()
    implot = ax.plot_surface(x, y, u_pred[i], cmap='viridis', edgecolor='none')
    ax.set_zlim(-0.5, 0.5)
    return implot,


ani = animation.FuncAnimation(fig, update,
                              frames=t_inp.shape[0],
                              fargs=(ax, fig), interval=100)
plt.show()
