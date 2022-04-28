import os

if __name__ == '__main__':
    for log in os.listdir('./log'):
        f = open('./log/' + log, 'r', encoding='utf-8')
        line = f.readline()
        cum_acc, cum_auprc = 0., 0.
        while line:
            idx = line.index('{')
            cls = line[:idx-1]
            acc_dic = eval(line[idx:])
            # if acc['acc'] < 0.5:
            #     print(log[:-4], cls)
            cum_acc += acc_dic['acc']
            cum_auprc += acc_dic['auprc']
            line = f.readline()
        
        print(log[:-4], cum_acc / 128, cum_auprc / 128)
        