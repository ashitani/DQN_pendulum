# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

d_good   = pd.read_csv("log_goodexp.csv",header=-1)
d_simple = pd.read_csv("log_simpleexp.csv",header=-1)

plt.figure(figsize=(6, 4))
plt.subplot(211)
plt.plot(d_good[0],d_good[1])
plt.hold(True)
plt.plot(d_good[0],d_good[5],"r")
plt.xlim(0,3e4)
plt.ylim(-400,1200)
plt.ylabel("Best ER",fontsize=8)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.subplot(212)
plt.plot(d_simple[0],d_simple[1])
plt.plot(d_simple[0],d_simple[5],"r")
plt.xlim(0,3e4)
plt.ylim(-400,1200)
plt.ylabel("Simple ER",fontsize=8)
plt.xlabel("Iteration",fontsize=8)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.savefig("compare.png")

plt.cla()
plt.figure(figsize=(6, 2))

plt.subplot(111)

plt.plot(d_good[0],d_good[1])
plt.hold(True)
plt.plot(d_good[0],d_good[5],"r")
plt.xlim(0,3e4)
plt.ylim(-400,1200)
plt.yticks(fontsize=6)
plt.xticks(fontsize=6)
plt.ylabel("Total rewards",fontsize=8)
plt.xlabel("Iteration",fontsize=8)
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig("bestER.png")

