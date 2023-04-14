from gunfolds.utils import zickle as zkl
from gunfolds.utils import bfutils
from matplotlib import pyplot as plt
from os import listdir
plt.style.use('seaborn-whitegrid')
CAPSIZE = 10000

density_dict = [0.2,0.25,0.3]
degree_lsit = [0.9,2,3,5]
undersampling_dict = [2,3,4]

l_py_sup = listdir('./res_CAP_'+str(CAPSIZE)+'_/old/pyRASL')
l_py_sup.sort()
if l_py_sup[0].startswith('.'):
    l_py_sup.pop(0)
gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/old/pyRASL/'+item) for item in l_py_sup]
py_list_sup = [item for sublist in gis for item in sublist]

l_py = listdir('./res_CAP_'+str(2000)+'_/pyRASL')
l_py.sort()
if l_py[0].startswith('.'):
    l_py.pop(0)
gis = [zkl.load('./res_CAP_'+str(2000)+'_/pyRASL/'+item) for item in l_py]
py_list = [item for sublist in gis for item in sublist]

l_cRASL = listdir('./myres_CAP_'+str(CAPSIZE)+'_/crasl')
l_cRASL.sort()
if l_cRASL[0].startswith('.'):
    l_cRASL.pop(0)
gis = [zkl.load('./myres_CAP_'+str(CAPSIZE)+'_/crasl/'+item) for item in l_cRASL]
c_list = [item for sublist in gis for item in sublist]

l_mslRASL = listdir('./res_CAP_'+str(CAPSIZE)+'_/old/msl_rasl')
l_mslRASL.sort()
if l_mslRASL[0].startswith('.'):
    l_mslRASL.pop(0)
gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/old/msl_rasl/'+item) for item in l_mslRASL]
msl_list = [item for sublist in gis for item in sublist]

l_dRASL = listdir('./myres_CAP_'+str(CAPSIZE)+'_/drasl')
l_dRASL.sort()
if l_dRASL[0].startswith('.'):
    l_dRASL.pop(0)
gis = [zkl.load('./myres_CAP_'+str(CAPSIZE)+'_/drasl/'+item) for item in l_dRASL]
d_list = [item for sublist in gis for item in sublist]

if __name__ == '__main__':
    pyRASL_sol ={}
    for item in py_list:
        if len(item['solutions']['eq']) ==2000:
            item = py_list_sup.pop()
        key = str(bfutils.g2num(item['gt']))
        if not len(item['solutions']['eq']) ==0:
            set = item['solutions']['eq']
            for e in set:
                value = {(e,item['u'])}
                try:
                    pyRASL_sol[key].append(value)
                except KeyError:
                    pyRASL_sol[key] = [value]

    dRASL_sol = {}

    for item in d_list:
        key = str(bfutils.g2num(item['gt']))
        if not len(item['solutions']['eq']) == 0:
            set = item['solutions']['eq']
            for e in set:
                value = {(e[0], item['u'])}
                try:
                    dRASL_sol[key].append(value)
                except KeyError:
                    dRASL_sol[key] = [value]

    cRASL_sol = {}


    for item in c_list:
        key = str(bfutils.g2num(item['gt']))
        if not len(item['solutions']['eq']) == 0:
            set = item['solutions']['eq']
            for e in set:
                value = {(e[0], item['u'])}
                try:
                    cRASL_sol[key].append(value)
                except KeyError:
                    cRASL_sol[key] = [value]


    mslRASL_sol = {}

    for item in msl_list:
        key = str(bfutils.g2num(item['gt']))
        if not len(item['solutions']['eq'] )== 0:
            set = item['solutions']['eq']
            for e in set:
                value = {(e[0], item['u'])}
                try:
                    mslRASL_sol[key].append(value)
                except KeyError:
                    mslRASL_sol[key] = [value]


    print ('check time')
    for key in pyRASL_sol.keys():
        dict_sol = {}
        sols = pyRASL_sol[key]
        for i in range(len(sols)):
            for val in sols[i]:
                try:
                    dict_sol[val[1]].append(val[0])
                    dict_sol[val[1]] = list(dict.fromkeys(dict_sol[val[1]]))
                except KeyError:
                    dict_sol[val[1]] = [val[0]]

        pyRASL_sol[key] = dict_sol

    for key in cRASL_sol.keys():
        dict_sol = {}
        sols = cRASL_sol[key]
        for i in range(len(sols)):
            for val in sols[i]:
                try:
                    dict_sol[val[1]].append(val[0])
                    dict_sol[val[1]] = list(dict.fromkeys(dict_sol[val[1]]))
                except KeyError:
                    dict_sol[val[1]] = [val[0]]

        cRASL_sol[key] = dict_sol

    for key in dRASL_sol.keys():
        dict_sol = {}
        sols = dRASL_sol[key]
        for i in range(len(sols)):
            for val in sols[i]:
                try:
                    dict_sol[val[1]].append(val[0])
                    dict_sol[val[1]] = list(dict.fromkeys(dict_sol[val[1]]))
                except KeyError:
                    dict_sol[val[1]] = [val[0]]

        dRASL_sol[key] = dict_sol

    for key in mslRASL_sol.keys():
        dict_sol = {}
        sols = mslRASL_sol[key]
        for i in range(len(sols)):
            for val in sols[i]:
                try:
                    dict_sol[val[1]].append(val[0])
                    dict_sol[val[1]] = list(dict.fromkeys(dict_sol[val[1]]))
                except KeyError:
                    dict_sol[val[1]] = [val[0]]

        mslRASL_sol[key] = dict_sol


    count = 0
    count2 = 0
    count3 = 0
    for k in mslRASL_sol:
        print ('check time')
        c_eq_size = [len(mslRASL_sol[k][i]) for i in mslRASL_sol[k].keys()]
        py_eq_size = [len(pyRASL_sol[k][i]) for i in pyRASL_sol[k].keys()]
        c_eq_size.sort()
        py_eq_size.sort()
        if max(py_eq_size + c_eq_size) >= CAPSIZE:
            print ("equivalant class is bigger than CAPSIZE. Not comparing")
            continue
        else:
            if py_eq_size == c_eq_size:
                print ("aligned")
            else:
                print ("differen euivalant class size")

                for u_rate in range(2,len(c_eq_size)+2):
                    if py_eq_size[u_rate-2] == c_eq_size[u_rate-2]:
                        continue
                    else:
                        g_true = bfutils.num2CG(int(k), 6)
                        gn_true = bfutils.undersample(g_true, u_rate)
                        # gn_true = bfutils.all_undersamples(g_true, u_rate)
                        li_dif = [i for i in mslRASL_sol[k][u_rate] + pyRASL_sol[k][u_rate] if
                                  i not in mslRASL_sol[k][u_rate] or i not in pyRASL_sol[k][u_rate]]
                        for item in li_dif:
                            # g_check = bfutils.all_undersamples(bfutils.num2CG(item,6))
                            lis_g_check = [bfutils.g2num(i) for i in bfutils.all_undersamples(bfutils.num2CG(item,6))]
                            if bfutils.g2num(gn_true) in lis_g_check:
                                # print ("the new answer in clingo set is correct")
                                count = count +1
                            else:
                                print ("this new answer that is in clingo is wrong")

    for k in dRASL_sol:
        print ('check time')
        c_eq_size = [len(dRASL_sol[k][i]) for i in dRASL_sol[k].keys()]
        py_eq_size = [len(pyRASL_sol[k][i]) for i in pyRASL_sol[k].keys()]
        c_eq_size.sort()
        py_eq_size.sort()
        if max(py_eq_size + c_eq_size) >= CAPSIZE:
            print ("equivalant class is bigger than CAPSIZE. Not comparing")
            continue
        else:
            if py_eq_size == c_eq_size:
                print ("aligned")
            else:
                print ("differen euivalant class size")

                for u_rate in range(2,len(c_eq_size)+2):
                    if py_eq_size[u_rate-2] == c_eq_size[u_rate-2]:
                        continue
                    else:
                        g_true = bfutils.num2CG(int(k), 6)
                        gn_true = bfutils.undersample(g_true, u_rate)
                        # gn_true = bfutils.all_undersamples(g_true, u_rate)
                        li_dif = [i for i in dRASL_sol[k][u_rate] + pyRASL_sol[k][u_rate] if
                                  i not in dRASL_sol[k][u_rate] or i not in pyRASL_sol[k][u_rate]]
                        for item in li_dif:
                            # g_check = bfutils.all_undersamples(bfutils.num2CG(item,6))
                            lis_g_check = [bfutils.g2num(i) for i in bfutils.all_undersamples(bfutils.num2CG(item,6))]
                            if bfutils.g2num(gn_true) in lis_g_check:
                                # print ("the new answer in clingo set is correct")
                                count = count +1
                            else:
                                print ("this new answer that is in clingo is wrong")

    for k in cRASL_sol:
        print ('check time')
        c_eq_size = [len(cRASL_sol[k][i]) for i in cRASL_sol[k].keys()]
        py_eq_size = [len(pyRASL_sol[k][i]) for i in pyRASL_sol[k].keys()]
        c_eq_size.sort()
        py_eq_size.sort()
        if max(py_eq_size + c_eq_size) >= CAPSIZE:
            print ("equivalant class is bigger than CAPSIZE. Not comparing")
            continue
        else:
            if py_eq_size == c_eq_size:
                print ("aligned")
            else:
                print ("differen euivalant class size")

                for u_rate in range(2,len(c_eq_size)+2):
                    if py_eq_size[u_rate-2] == c_eq_size[u_rate-2]:
                        continue
                    else:
                        g_true = bfutils.num2CG(int(k), 6)
                        gn_true = bfutils.undersample(g_true, u_rate)
                        # gn_true = bfutils.all_undersamples(g_true, u_rate)
                        li_dif = [i for i in cRASL_sol[k][u_rate] + pyRASL_sol[k][u_rate] if
                                  i not in cRASL_sol[k][u_rate] or i not in pyRASL_sol[k][u_rate]]
                        for item in li_dif:
                            # g_check = bfutils.all_undersamples(bfutils.num2CG(item,6))
                            lis_g_check = [bfutils.g2num(i) for i in bfutils.all_undersamples(bfutils.num2CG(item,6))]
                            if bfutils.g2num(gn_true) in lis_g_check:
                                # print ("the new answer in clingo set is correct")
                                count = count +1
                            else:
                                print ("this new answer that is in clingo is wrong")

    print ("finish")