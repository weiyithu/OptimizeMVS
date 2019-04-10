import os
import utils

diary_list = os.listdir('diary')

for fname in diary_list:
    with open(os.path.join('diary', fname)) as file_:
        lines = file_.readlines()
        if len(lines) == 0:
            continue
        if 'EMD' in lines[-1]:
            mode = 'EMD'
        else:
            mode = 'CD_fps'
        test1, test2, it_best = utils.report_best(os.path.join('diary', fname), mode)
        task = lines[0].strip('\n')
        if test1 == -1:
            continue 
        if task == 'test':
            continue
        it = int((lines[-1].split('_it')[1]).split('.ckpt')[0])
        print "name\t\t\ttask\t\t\t\titer\tmax\t"+mode+"\tCD"
        print "{0:20s}\t{1:30s}\t{2:06d}\t{5:06d}\t{3:.2f}\t{4:.2f}".format(fname, task, it, test1, test2, it_best)

