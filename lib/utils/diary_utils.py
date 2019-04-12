import string

# ------ view dairy function ------
def parse_report(diary_path, key_):
    timeline = []
    nameline = []
    idx = 0
    with open(diary_path) as file:
        flag = 2
        lines = file.readlines()
        for line in lines:
            if flag == 2:
                if line.strip('\n') == 'loss_diary':
                    flag = 2
                else: flag = 1
                continue
            if flag == 1:
                line = line.strip('\n')
                info = string.split(line, ',')
                key_list = []
                for str_ in info:
                    key_list.append(string.split(str_, ':')[0])
                for i in range(len(key_list)):
                    key = key_list[i]
                    if key == key_:
                        idx = i
                flag = 0

            info = string.split(line, ',')
            timeline.append(float(string.split(info[idx], ':')[1]))
            nameline.append(int((line.split('_it')[1]).split('.ckpt')[0])) 
    return timeline,nameline

def report_best(diary_path, mode):
    if mode == 'EMD':
        emd_timeline, _ = parse_report(diary_path, 'EMD_test')
        cd_timeline, nameline = parse_report(diary_path, 'CD_test')
        if not len(emd_timeline):
            return -1, -1, -1
        else: 
            emd = min(emd_timeline)
            cd = min(cd_timeline)
            return emd, cd, nameline[np.argmin(cd_timeline)]
    else:
        cd_fps_timeline, nameline = parse_report(diary_path, 'CD_fps_test')
        cd_timeline, _ = parse_report(diary_path, 'CD_test')
        if not len(cd_fps_timeline):
            return -1, -1, -1
        else: 
            cd_fps = min(cd_fps_timeline)
            cd = min(cd_timeline)
            return cd_fps, cd, nameline[np.argmin(cd_fps_timeline)]


