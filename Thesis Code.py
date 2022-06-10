#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hfda
import numpy as np
import scipy.integrate as si
import scipy.interpolate as so
import scipy.constants as sc
import scipy.stats as ss
import matplotlib.pyplot as plt
import sympy as sm
import plotly.graph_objs as go
import pandas as pd
import time


# In[2]:


def ring(radius, location):
    alpha, beta, gamma, x0, y0, z0 = location
    phi = np.linspace(0, 2*np.pi, 100)
    xring = np.cos(alpha)*np.cos(beta)*np.cos(phi) + (np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma))*np.sin(phi)
    yring = np.sin(alpha)*np.cos(beta)*np.cos(phi) + (np.cos(alpha)*np.cos(gamma)+np.sin(alpha)*np.sin(beta)*np.sin(gamma))*np.sin(phi)
    zring = -1*np.sin(beta)*np.cos(phi) + np.cos(beta)*np.sin(gamma)*np.sin(phi)
    xring *= radius
    yring *= radius
    zring *= radius
    return (xring+x0, yring+y0, zring+z0)


# In[3]:


def rotation(angles):
    alpha, beta, gamma, x0, y0, z0 = angles
    matrix = np.asarray([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.sin(beta)*np.cos(gamma)],
                      [np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(gamma)+np.sin(alpha)*np.sin(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
                      [-1*np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    return matrix


# In[4]:


#magnetic field, no interpolation
def rkmagfield(radius, current):
    t, x, y, z = sm.symbols('t, x, y, z')
    l = sm.Matrix([radius*sm.cos(t), radius*sm.sin(t), 0])
    r = sm.Matrix([x,y,z])
    rprime = r-l
    rphat = rprime/rprime.norm()
    inside = sm.diff(l,t).cross(rphat)/(rprime.norm()**2)
    intx = sm.lambdify([t,x,y,z], inside[0])
    inty = sm.lambdify([t,x,y,z], inside[1])
    intz = sm.lambdify([t,x,y,z], inside[2])
    field = lambda point: ((sc.mu_0*current)/(4*np.pi)) * np.array([si.quad(intx,0,2*np.pi,args=(point[0], point[1], point[2]))[0], si.quad(inty,0,2*np.pi,args=(point[0], point[1], point[2]))[0], si.quad(intz,0,2*np.pi,args=(point[0], point[1], point[2]))[0]])
    return field


# In[5]:


#interpolation evaluator for rk step
def newnetdir(func, rings, point, back):
    total = np.asarray(back)
    for p in rings:
        orgpoint = point - np.asarray(p)[3:6]
        R = rotation(p)
        orgpoint = np.matmul(R.T, orgpoint.T).T
        field = np.asarray(func(orgpoint))
        if np.any(np.isnan(field)) == True:
            return False, False
        field = np.matmul(R, field).flatten()
        total = total + field
    norm = np.sqrt(total[0]**2 + total[1]**2 + total[2]**2)
    direction = total/norm
    return direction


# In[6]:


#runge-kutta 4 step
def steprk(func, rings, point, h, back):
    original = point
    dir1 = newnetdir(func, rings, point, back)
    if type(dir1) == bool:
        return False
    k1 = h*dir1
    point = original + k1/2
    dir2 = newnetdir(func, rings, point, back)
    if type(dir2) == bool:
        return False
    k2 = h*dir2
    point = original + k2/2
    dir3 = newnetdir(func, rings, point, back)
    if type(dir3) == bool:
        return False
    k3 = h*dir3
    point = original + k3
    dir4 = newnetdir(func, rings, point, back)
    if type(dir4) == bool:
        return False
    k4 = h*dir4
    nextpoint = original + k1/6 + k2/3 + k3/3 + k4/6
    return nextpoint


# In[7]:


#RK step closed field line
def rkfieldline(function, rings, startpoint, stepsize, limit=2e5):
    nowpoint = startpoint
    newpoint = np.asarray([0,0,-1])
    line = startpoint
    iteration = 0
    difference = 1
    dif2 = -1
    dif1 = -1
    #while iteration < limit and difference >= stepsize:
    closeapp = False
    while closeapp == False and iteration < limit:
        if (iteration % 10000) == 0:
            print(iteration)
        newpoint = steprk(function, rings, nowpoint, stepsize, [0,0,0])
        if type(newpoint) == bool:
            if newpoint == False:
                return line, nowpoint, -1, iteration
        line = np.vstack([line, newpoint])
        lastpoint = nowpoint
        nowpoint = newpoint
        dif2 = dif1
        dif1 = difference
        difference = np.sqrt((line[0,0]-nowpoint[0])**2 + (line[0,1]-nowpoint[1])**2 + (line[0,2]-nowpoint[2])**2)
        if ((dif2 > dif1) and (difference > dif1) and (iteration > 2)):
            nowpoint = lastpoint
            difference = dif1
            closeapp = True
            break
        iteration += 1
    diff = nowpoint - line[0, ::]
    return line, diff, difference, iteration


# In[8]:


#RK step field line full
def rktrace(function, rings, startpoint, stepsize, limit=2e5, background = [0,0,0]):
    nowpoint = startpoint
    newpoint = np.asarray([0,0,-1])
    line = startpoint
    iteration = 0
    difference = 1
    while iteration < limit:
        if (iteration % 1000) == 0:
            print(iteration)
        newpoint = steprk(function, rings, nowpoint, stepsize, background)
        if type(newpoint) == bool:
            if newpoint == False:
                return line, nowpoint, -1, iteration
        line = np.vstack([line, newpoint])
        lastpoint = nowpoint
        nowpoint = newpoint
        difference = np.sqrt((line[0,0]-nowpoint[0])**2 + (line[0,1]-nowpoint[1])**2 + (line[0,2]-nowpoint[2])**2)
        iteration += 1
    diff = nowpoint - line[0, ::]
    return line, diff, difference, iteration


# In[9]:


R = 1
I = 10
p1 = [0,0,0,0,0,0]
p2 = [0,0,np.pi/2,3,0,0]
p3 = [0,0,np.pi/2,1,0,0]
loop1 = ring(R, p1)
loop2 = ring(R, p2)
loop3 = ring(R, p3)


# In[10]:


def lyapunov(traj1, traj2):
    x1 = traj1[::,0]
    y1 = traj1[::,1]
    z1 = traj1[::,2]
    x2 = traj2[::,0]
    y2 = traj2[::,1]
    z2 = traj2[::,2]
    diffx, diffy, diffz = x1-x2, y1-y2, z1-z2
    sep = np.sqrt(np.power(diffx,2) + np.power(diffy,2) + np.power(diffz,2))
    isep = sep[0]
    lsep = np.log(sep)
    iters = np.arange(len(sep)) + 1
    lia = np.log(sep/isep)/iters
    return sep, iters, lia, lsep


# In[11]:


def quants(s, N=1e4):
    hfd = []
    kmaxmax = int(np.floor(N/2))
    for k in [2,kmaxmax]:
        h = hfda.measure(s,k)
        hfd.append(h)
        print(k, h)
    hfdev = h - hfd[0]
    temp=0
    var = s[1]-s[0]
    count=0
    for i in range(1,len(s)):
        prev = var
        var = s[i]-s[i-1]
        psign = prev/np.abs(prev)
        sign = var/np.abs(var)
        if psign != sign:
            count += 1
        temp = temp+np.abs(var)
    temp = temp/(np.amax(s)-np.amin(s))
    print(temp,count)
    return hfdev, temp, count


# In[12]:


def strength(func, rings, point, back):
    total = np.asarray(back)
    for p in rings:
        orgpoint = point - np.asarray(p)[3:6]
        R = rotation(p)
        orgpoint = np.matmul(R.T, orgpoint.T).T
        field = np.asarray(func(orgpoint))
        if np.any(np.isnan(field)) == True:
            return False, False
        field = np.matmul(R, field).flatten()
        total = total + field
    norm = np.sqrt(total[0]**2 + total[1]**2 + total[2]**2)
    return norm

