import numpy as np
import pandas as pd
import math
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
from gurobipy import *
import random


def factorial(x):
    if x == 0:
        return 1
    else:
        y = 1
        for i in range(1, (x + 1)):
            y = i * y
        return y


# list
def q_func(n, mar):
    p = mar / n / sr
    pn = mar / sr
    pn_n = np.array([math.pow(pn[t], n[t]) for t in range(T)])
    f_n = np.array([factorial(int(n[t])) for t in range(T)])


    # w1 = (pn_n*p)/u/((1-p)**2)/f_n
    # w2 = np.array([sum([math.pow(pn[t], k) / factorial(k) for k in range(int(n[t]))]) for t in range(T)])
    # w3 = pn_n / f_n / (1 - p)


    ls1 = p / ((1 - p) ** 2) / f_n * pn_n
    ls2 = np.array([sum([math.pow(pn[t], k) / factorial(k) for k in range(int(n[t]))]) for t in range(T)])
    ls3 = pn_n / f_n / (1 - p)
    ls4 = pn

    ls = ls1 / (ls2 + ls3) + ls4
    return ls / mar - 1 / sr


def fitness_func(vars_x):
    s, e, n = split(vars_x)
    u, q = uq_creat(n)
    a = np.array([0] + [min(q[t - 1], u[t]) for t in range(1, T)])
    b = u - a

    # e_adj = np.concatenate((e, e))
    #
    # # rhs-lhs <0 不满足约束 工作时长上界， error1=1
    # error1 = 1 - int(np.min(np.array([np.sum(e_adj[t + 1:t + UB + 1]) for t in range(T)]) - n) >= 0)
    # # lhs - rhs <0 不满足约束 工作时长下界， error2=1
    # error2 = 1 - int(np.min(n - s - np.array([np.sum(e_adj[t + 1:t + LB]) for t in range(T)])) >= 0)
    #
    # # true = 1
    # error3 = 1 - int(np.max(n) <= N and np.min(n) >= 1)
    #
    # #
    # error4 = np.sum(np.where(q < 0, 1, 0))

    # objective function

    w1 = np.where(a > n, (a - n) * (a - n + 1) / (2 * n * sr), 0)
    w2 = b * q_func(n, u)
    w3 = np.where(q < ar, q * (q + 1) / 2 / ar, q + (1 - ar) / 2)

    TP = np.sum(w1 + w2 + w3)
    TD = np.sum(n)
    obj = TP + 2 * TD
    return obj


def split(vars_x):
    s = vars_x[:T]
    e = vars_x[T:2 * T]
    n = vars_x[2 * T:3 * T]
    return s, e, n


def combine(s, e, n):
    return np.concatenate((s, e, n))




def work_time(vars_x, pop):
    s, e, n = split(vars_x)
    s_list = np.array([np.sum(s[:t + 1]) for t in range(T)])
    st = np.where(pop <= s_list)[0][0]
    e_adj = np.append(e[EST+2:],[e[EST+1]],axis=0)
    e_list = np.array([np.sum(e_adj[:t + 1]) for t in range(T - EST - 1)])
    et = np.where(pop <= e_list)[0][0]
    et = (et + 1) % (T - EST - 1) + EST + 1
    gap = (et - st + 24) % T
    return st, et,gap


# 得到一个N的解
def ini_gurobi():
    # EST = 6 FST=0 在模型中0代表24，一天为0-23点
    model = Model()

    # variable definition
    s = np.array([model.addVar(lb=0, ub=N, vtype=GRB.INTEGER, name="s2[%d]" % t) for t in range(T)])
    e = np.array([model.addVar(lb=0, ub=N, vtype=GRB.INTEGER, name="e2[%d]" % t) for t in range(T)])
    u = np.array([model.addVar(lb=1, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="u2[%d]" % t) for t in range(T)])


    a = np.array([model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a2[%d]" % t) for t in range(T)])
    b = np.array([model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b2[%d]" % t) for t in range(T)])
    c = np.array([model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c2[%d]" % t) for t in range(T)])

    # 得到n,q
    n = n_creat(s, e)

    # constraints
    model.addConstr(np.sum(s[FST:EST+1]) == 0)
    model.addConstr(np.sum(e[FST:EST+1]) == 0)
    model.addConstr(np.sum(s) <= N)
    model.addConstr(np.sum(s) == np.sum(e))
    model.addConstrs((1 <= n[t] <= N) for t in range(T))

    e_tmp = np.concatenate((e, e))
    model.addConstrs((n[t] <= np.sum(e_tmp[t + 1:t + UB + 1])) for t in range(T))
    model.addConstrs((n[t] - s[t] >= np.sum(e_tmp[t + 1:t + LB])) for t in range(T))

    q = q_creat(u)
    model.addConstrs((a[t] == n[t] * sr) for t in range(T))
    model.addConstr(b[0] == ar[0])
    model.addConstrs((b[t] == ar[t] + q[t - 1]) for t in range(1, T))

    model.addConstrs((c[t] == min_(a[t], b[t])) for t in range(T))
    model.addConstrs((u[t] <= c[t]) for t in range(T))
    model.addConstrs((u[t] >= c[t] - 1) for t in range(T))
    model.addConstrs((q[t] >= 0) for t in range(T))

    model.setObjective(np.sum(s), GRB.MAXIMIZE)

    model.optimize()
    if model.status == GRB.Status.OPTIMAL:
        var = model.getVars()
        s = np.array([var[j].getAttr("Xn") for j in range(T)])
        e = np.array([var[j].getAttr("Xn") for j in range(T, T * 2)])
        n = n_creat(s, e)
        return np.concatenate((s, e, n))


def vns(vars_x):
    count = 0
    best_solution,best_value = vdn(vars_x)
    while count <= max_iterations:
        candidate = shaking(vars_x)
        current_solution, current_value = vdn(candidate)
        count+=1
        if current_value < best_value:
            best_value = current_value
            best_solution = np.copy(current_solution)

            print("count",count,best_value)

    return best_solution, best_value


# 随机减少一个
def shaking(vars_x):
    print("shaking start ----------------------")
    s, e, n = split(vars_x)
    best_value = 10000
    # 选择每个医生,pop表示第几个

    for i in range(3):
        pop_size = np.sum(s)
        if pop_size <= 3:
            return vars_x
        else:
            i = 0
            j = 0
            while i == 0 :
                current_s = np.copy(s)
                current_e = np.copy(e)
                pop = np.random.randint(pop_size - 1)
                st, et,gap = work_time(vars_x, pop + 1)
                # et = (st + gap) % T
                current_s[st] -= 1
                current_e[et] -= 1
                n = n_creat(current_s, current_e)
                i = np.min(n)
            s = current_s
            e = current_e

    print(s)
    print("shaking end--------------")
    return combine(s,e,n)


# 随机增加一名医生
def local_search0(best_solution, best_value):
    print("local_search0 start--------------")
    s, e, n = split(best_solution)
    if np.sum(s) < N:
        current_s = np.copy(s)
        current_e = np.copy(e)
        # np.random.randint(EST, T - EST + 1)= 6-18 -> +10-> 16-28 -> %17 ->16 ~ 0 -11-> +7-> 23~7-18
        st = (np.random.randint(EST, T - EST+1)+10)%17+7
        current_s[st] += 1
        gap = np.random.randint(LB, UB + 1)
        et = (st + gap) % T
        while et <= EST:
            gap = (gap - LB + 1) % 4 + LB
            et = (st + gap) % T
        current_e[et] += 1
        n = n_creat(current_s, current_e)
        current_solution = combine(current_s, current_e, n)
        current_value = fitness_func(current_solution)
        if current_value < best_value:
            best_value = current_value
            best_solution = np.copy(current_solution)
    print("local_search0 end----------")
    return best_solution, best_value


# 随机选择一名医生，上班时间延后，工作时间不变
def local_search1(best_solution, best_value):
    print("local_search1 start--------------")

    s, e, n = split(best_solution)

    # 选择每个医生,pop表示第几个
    pop_size = int(np.sum(s))

    for i in range(pop_size - 1):
        current_s = np.copy(s)
        current_e = np.copy(e)
        st,et, gap = work_time(best_solution, i + 1)
        current_st = (st + 1) % T
        current_et = (st + 1 + gap) % T
        current_s[st] -= 1
        current_e[et] -= 1
        current_s[current_st] += 1
        current_e[current_et] += 1
        current_n = n_creat(current_s, current_e)
        if current_et <= EST or current_st <= EST or np.min(current_n) == 0:
            pass
        else:
            current_solution = combine(current_s, current_e, current_n)
            current_value = fitness_func(current_solution)
            if current_value < best_value:
                best_value = current_value
                best_solution = current_solution
            else:
                pass

    print("local_search1 end--------------")
    return best_solution, best_value


# 随机选择一名医生，增加上班时间时长
def local_search2(best_solution, best_value):
    print("local_search2 start--------------")
    s, e, n = split(best_solution)

    # 选择每个医生,pop表示第几个
    pop_size = int(np.sum(s))
    for i in range(pop_size-1):
        st, et, gap = work_time(best_solution, i + 1)
        for j in range(3):
            current_e = np.copy(e)
            gap = (gap - LB + 1) % 4 + LB
            current_et = (st + gap) % T
            current_e[et] -= 1
            current_e[current_et] += 1
            current_n = n_creat(s, current_e)
            if current_et <= EST or np.min(current_n) == 0:
                pass
            else:
                current_solution = combine(s, current_e, current_n)
                current_value = fitness_func(current_solution)
                if current_value < best_value:
                    best_value = current_value
                    best_solution = np.copy(current_solution)
                else:
                    pass

    print("local_search2 end--------------")
    return best_solution, best_value


def vdn(vars_x):
    best_solution = vars_x
    best_value = fitness_func(vars_x)
    # 随机增加一个医生
    i =1
    while (i!=0):
        i=0
        solution = best_solution
        value = best_value

        current_solution, current_value = local_search0(solution, value)
        if current_value < best_value:
            best_value = current_value
            best_solution = np.copy(current_solution)
            i +=1
            print (best_solution[:T])
            print ("search0_",current_value)

        current_solution, current_value = local_search1(solution, value)
        if current_value < best_value:
            best_value = current_value
            best_solution = np.copy(current_solution)
            i+=1
            print(best_solution[:T])
            print("search1_",current_value)

        current_solution, current_value = local_search2(solution, value)
        if current_value < best_value:
            best_value = current_value
            best_solution = np.copy(current_solution)
            i+=1
            print(best_solution[:T])
            print ("search2_",current_value)

    return best_solution, best_value


def n_creat(s, e):
    n1 = np.array([np.sum(s[1:t + 1] - e[1:t + 1]) + e[EST + 1] for t in range(1, T)])
    n = np.insert(n1, 0, e[EST + 1])
    return n


def q_creat(u):
    q = np.array([np.sum(ar[:t + 1] - u[:t + 1]) for t in range(T)])
    return q


def uq_creat(n):

    ar0 = np.array([np.floor(i) for i in ar])
    u = np.zeros(T)
    q = np.zeros(T)
    u[0] = min(np.floor(n[0] * sr-para),ar0[0])
    q[0] = ar0[0] - u[0]
    for t in range(1, T):
        u[t] = min(q[t - 1]+ar0[t], np.floor(n[t] * sr-para))
        q[t] = np.sum(ar0[:t + 1] - u[:t + 1])
    return u, q


def np_to_int_list(n):
    return [int(n[t]) for t in range(len(n))]


def final(pop_x, fitness):
    vars_x = pop_x[np.argsort(fitness)[0]]
    obj = np.sort(fitness)[0]
    return vars_x, obj


def plot(values):
    values = np.array(values).T
    cmap = plt.cm.coolwarm
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
    fig, ax = plt.subplots()
    lines = ax.plot(values)
    ax.legend(lines)
    plt.show()
    return 0


T = 24
N = 10
# forbid start time / end time
FST = 0
EST = 6
UB = 8
LB = 5
P = 100
L0 = 0
error = 0.01
sr = 5.9113
para = 1.7

max_iterations = 1000
writer = pd.ExcelWriter('d:\\Users\\royce\\Desktop\\工工综合实验\\m1.xlsx')

# np.random.seed(100)

"""read data from 服务性系统问题-问题数据.xlsx"""
IO = 'd:\\Users\\royce\\Desktop\\工工综合实验\\服务系统问题-问题数据.xlsx'
df_ar = pd.read_excel(io=IO, header=None, index_col=None)
ar = np.array(list(df_ar.ix[6, 1:]))
print (ar)

if __name__ == '__main__':
    # 初始化
    vars_x = ini_gurobi()
    current_solution,current_value = vdn(vars_x)

    best_solution, best_value = vns(current_solution)

    print ("final", best_value)
    s, e, n = split(best_solution)
    u, q = uq_creat(n)


    df = pd.DataFrame({'s': pd.Series(s), 'e': pd.Series(e), 'n': pd.Series(n),
                       'u': pd.Series(u),'q':pd.Series(q)})
    df.to_excel(writer, header=True, index=True)

    writer.save()













