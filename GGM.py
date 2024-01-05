# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:59:20 2024

@author: Dr. Lianyi Liu  , email (mytruth@126.com; liu1996@nuaa.edu.cn)
"""

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from User_defined_Functions import first_accumulation, _equation_str, _print_error,replace_string
from scipy.integrate import odeint
import graphviz
import sympy as sp
import matplotlib.pyplot as plt

class Generalized_grey_model(object):
    def __init__(self,df,nf):
        self.df=df
        self.nf=nf
    def generate_dataset(self):
        [t,y]=[self.df['time'],self.df['data']]
        self.t=np.array(t)
        self.y=np.array(y)
        self.n=y.shape[0]
        self.x_1, self.t=first_accumulation(self.y.reshape(-1), self.t.reshape(-1), self.n)
        datax=np.array(list(zip(self.t,self.x_1)))
        self.train_size = int(len(datax) -self.nf)
        self.test_size = len(datax)-self.train_size
        self.x_train = datax[0:self.train_size]
        self.y_train = self.y[0:self.train_size]
        self.x_test = datax[self.train_size:]
        self.y_test = self.y[self.train_size:]
    def structure_identification_SG(self,function_set= ['add', 'sub', 'mul', 'div'],parsimony_coefficient=0.01):
        self.generate_dataset()
        self.est_gp = SymbolicRegressor(population_size=2000,
                                   generations=20, stopping_criteria=0.01,
                                   function_set=function_set,
                                   const_range=(-1000,1000),
                                   feature_names=['t','x_1'],
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   max_samples=0.9, verbose=1,
                                   parsimony_coefficient=parsimony_coefficient, random_state=0)
        self.est_gp.fit(self.x_train, self.y_train)
        self.train_fit = self.est_gp.predict(self.x_train)
        self.test_fit = self.est_gp.predict(self.x_test)
        self.y_pre=np.concatenate((self.train_fit,self.test_fit))
        #print(self.est_gp._program)
        self.graphviz_pdf()
        self.equation_str=_equation_str(self.est_gp._program)
        self.print_equation()
    def f_equation(self,x_1,t):
            return eval(self.equation_str)
    def predict(self,t):
        # Define initial conditions
        y0 = self.x_1[0]
        # Using odeint functions to solve differential equations
        x_1_pre = odeint(self.f_equation, y0, t)
        self.y_pre2=np.concatenate((np.array([self.y[0]]),np.diff(x_1_pre.reshape(-1,))))
        return self.y_pre2
    def print_error(self):
        _print_error(self.y,self.y_pre2,self.train_size)
    def graphviz_pdf(self):
        dot_data = self.est_gp._program.export_graphviz()
        self.graph = graphviz.Source(dot_data)
        #self.graph.view() #Output to PDF file
    def print_equation(self):
        t = sp.symbols('t')
        x_1 = sp.Function('x_1')(t)
        self.equation=sp.Eq(x_1.diff(t),sp.sympify(replace_string(self.equation_str)))

if __name__ == '__main__':
    #generate model
    nf=4 #Test set size
    df=pd.read_excel('data.xlsx')
    GGM=Generalized_grey_model(df,nf)
    #training model
    function_set = ['add', 'sub', 'mul', 'div','log', 'sin','cos'] 
    parsimony_coefficient=0.07 #Complexity penalty coefficient
    GGM.structure_identification_SG(function_set,parsimony_coefficient)
    #predict
    y_predict=GGM.predict(GGM.t)
    plt.plot(GGM.t, GGM.y)
    plt.plot(GGM.t, GGM.y_pre2)
    plt.xlabel('time')
    plt.ylabel('data')
    plt.show()
    #error
    GGM.print_error()
    #Tree Formula Chart
    GGM.graph
    #equation
    GGM.equation
    sp.pprint(GGM.equation)


