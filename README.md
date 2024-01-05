# GGM

`GGM` is a Python package used for adaptively fitting unknown differential equations.

`GGM`  uses symbolic regression as a function fitting tool. `ODE` function is used as a tool for solving differential equations. `Sympy` and `graphviz` are served as visualization tools for functions.


`GGM` provides the following functionalities:

* Using supervised learning methods to fit the optimal structure of differential equations
* Libraries to compile Python source code to visualized formulas or tree charts.
* It can be used for fitting and predicting time series data tasks.

## Example
#generate model
```
nf=4 #Test set size
df=pd.read_excel('data.xlsx')
GGM=Generalized_grey_model(df,nf)
```
#training model
```
function_set = ['add', 'sub', 'mul', 'div','log', 'sin','cos'] 
parsimony_coefficient=0.07 #Complexity penalty coefficient
GGM.structure_identification_SG(function_set,parsimony_coefficient)
```
```
Out:
    |   Population Average    |             Best Individual              |
---- ------------------------- ------------------------------------------ ----------
 Gen   Length          Fitness   Length          Fitness      OOB Fitness  Time Left
   0    15.94      2.33544e+10        7          2.89244          3.43169     28.48s
   1    13.45      2.11638e+06        7          2.64555          5.65364     28.10s
   2    16.24      9.76109e+08       12          2.42225          2.17735     25.23s
   3     6.20          3977.46       11          2.12796           2.8791     20.11s
   4     3.38          5895.52       11          2.07857          3.32365     19.33s
   5     3.63           157349       11          2.11583           3.2799     17.03s
   6     4.10          58392.6       11          1.94678          4.80131     15.14s
   7     5.81           2616.5       11          1.99374          4.37871     14.82s
   8     6.81          9840.32       11          1.99468          4.37025     13.90s
   9     6.91          4936.27        7          1.97899          4.09582     12.44s
  10     6.98           749734        7          1.89627          4.84034     11.24s
  11     6.99          3171.81        9          1.83453           2.1049     10.04s
  12     7.05          33457.3        9          1.77531          2.63789      9.19s
  13     7.13      4.02057e+07        9          1.73695          2.98311      7.49s
  14     7.43      3.59203e+06        9          1.69269          3.38149      6.42s
  15     7.76      3.11129e+06        9          1.63878          3.86667      5.11s
  16     8.64      1.96188e+06        9          1.66078          3.66865      3.90s
  17     8.94          62619.7        9          1.64171          3.84034      2.59s
  18     8.95      2.10623e+06        9          1.64316          3.82724      1.28s
  19     8.88          4724.67        9          1.62767          3.96666      0.00s
```
#predict
```
y_predict=GGM.predict(GGM.t)
plt.plot(GGM.t, GGM.y)
plt.plot(GGM.t, GGM.y_pre2)
plt.xlabel('time')
plt.ylabel('data')
plt.show()
```
#error
```
GGM.print_error()
```
```
Out:
The fitting MAE 2.5978930314992494 
The fitting MAPE 0.032250991337976924 
The forecasting MAE 1.7402563881370483 
The forecasting RMSE 1.8150325900715465 
The forecasting MAPE 0.025416924676978404
```
#Tree Formula Chart
```
GGM.graph
```
#equation
```
GGM.equation
sp.pprint(GGM.equation)
```
```
Out:
d                                     t + x₁
──(x₁(t)) = -0.00260357001520485⋅x₁ + ──────
dt                                      t   
```
## Required Package Version
My package version is as follows
* Python 3.8.5
* gplearn 0.4.2
* scipy 1.8.1
* graphviz 0.20.1
* sympy 1.6.2

## Programming language
Python

## Contributing
This repository is contribution friendly.

## FAQs
If you have any questions, please send me an email to mytruth@126.com.





