# coding: utf-8

from dqn_pendulum import *

from PIL import Image,ImageDraw,ImageFont
import imageio
from matplotlib import pyplot as plt
import sys

def circle(draw, c, r, col):
    cx,cy=c
    draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=col)

def draw_pendulum(th,title):
    im = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(im)
    c=(100,100)
    l=80
    mc=(c[0]+l*np.sin(th),c[1]+l*np.cos(th))
    circle(draw, c,8, (0,0,0))
    circle(draw, c,5, (255,255,255))
    circle(draw, c,3, (0,0,0))
    draw.line((c[0],c[1],mc[0],mc[1]), fill=(0,0,0),width=3)
    draw.text((10,10),title,fill=(0,0,0))
    return np.asarray(im)

def log2gif(log,filename,title):
    writer = imageio.get_writer(filename, fps=30)
    for x in log:
        writer.append_data(draw_pendulum(x[0],title))
    writer.close()

def log2profile(log,filename):
    plt.figure(figsize=(6*0.75, 4*0.75))
    plt.subplot(211)
    plt.plot(-np.cos(np.asarray(log)[:,0]))
    plt.ylabel("Height",fontsize=10)
    plt.ylim(-1.2,1.2)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.subplot(212)
    plt.plot(np.asarray(log)[:,1])
    plt.ylim(-1.2,1.2)
    plt.ylabel("Torque",fontsize=10)
    plt.xlabel("Simulation step",fontsize=10)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.gcf().subplots_adjust(bottom=0.15,left=0.15,right=0.95)
    plt.savefig(filename)

if __name__ == '__main__':

    #####  plot profile
    argv=sys.argv
    if len(argv)!=2:
        print "Usage: python %s model_file" % (sys.argv[0], )

    filename = sys.argv[1]

    agent=DQNAgent()
    env=pendulumEnvironment()
    sim=simulator(env,agent)
    serializers.load_npz(filename, agent.model)
    total_reward,log=sim.run(train=False, movie=False,enableLog=True)
    log2profile(log,"profile.png")
    log2gif(log,"animation.gif", "")

    #####  plot dummyAgent

    # agent=dummyAgent()
    # env=pendulumEnvironment()
    # sim=simulator(env,agent)
    # total_reward,log=sim.run(train=False, movie=False,enableLog=True)
    # log2gif(log,"initial.gif", "")

    #####  plot all movies

    # import glob
    # import re

    # for m in glob.glob("model/*.model"):
    #     agent=DQNAgent()
    #     env=pendulumEnvironment()
    #     sim=simulator(env,agent)
    #     i=re.findall("\d+",m)[0]
    #     serializers.load_npz(m, agent.model)
    #     total_reward,log=sim.run(train=False, movie=False,enableLog=True)
    #     log2gif(log,"movie/%s.gif"% i, i)

