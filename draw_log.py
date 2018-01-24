import matplotlib.pyplot as plt


def draw_acc(filename, title):
    epochs = list()
    train_accs = list()
    vali_accs = list()
    with open(filename, 'r') as f:
        for line in f:
            if 'epoch' in line and 'train_acc' in line and 'vali_acc' in line:
                eles = line.split()
                epochs.append(int(eles[1]))
                train_accs.append(float(eles[4]))
                vali_accs.append(float(eles[10]))
    plt.plot(epochs, train_accs)
    plt.plot(epochs, vali_accs)
    plt.legend(['Training Set', 'Verification Set'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(title)
    # plt.ylim([0, 1])
    plt.show()


def draw_loss(filename, title):
    epochs = list()
    train_losses = list()
    vali_losses = list()
    with open(filename, 'r') as f:
        for line in f:
            if 'epoch' in line and 'train_loss' in line and 'vali_loss' in line:
                eles = line.split()
                epochs.append(int(eles[1]))
                train_losses.append(float(eles[7]))
                vali_losses.append(float(eles[13]))
    plt.plot(epochs, train_losses)
    plt.plot(epochs, vali_losses)
    plt.legend(['Training Set', 'Verification Set'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim([0, 1.2])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    draw_loss('./nn_cnn_rnn_3.log', 'cnn lstm loss')
    # draw_loss('./nn_cnn_wave_0.log', 'cnn wav loss')
