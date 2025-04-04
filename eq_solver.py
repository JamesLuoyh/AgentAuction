# from sympy import symbols, solve, diff, exp, lambdify, Eq
from scipy.optimize import fsolve, minimize
import numpy as np
from math import exp

c2=0.75#0.875
a2 = 2#c2+1

c1 = 1
a1 = c1 + 1

def obj1(x, y):
   return -(x-c1)*exp(4*(a1-x))/(exp(4*(a2-y)) + exp(4*(a1-x)) + 1)


def obj2(x, y):
    return -(y-c2)*exp(4*(a2-y))/(exp(4*(a2-y)) + exp(4*(a1-x)) + 1)


def optimize_iteratively(x0, y0, threshold=1e-10, max_iterations=1000):
    x, y = x0, y0
    for i in range(max_iterations):
        old_x, old_y = x, y

        # Optimize objective function 1
        res1 = minimize(lambda x_temp: obj1(x_temp, y), x, method='L-BFGS-B')
        x = res1.x[0]

        # Optimize objective function 2
        res2 = minimize(lambda y_temp: obj2(x, y_temp), y, method='L-BFGS-B')
        y = res2.x[0]

        # Check for convergence
        if np.sqrt((x - old_x)**2 + (y - old_y)**2) < threshold:
            break
    return x, y, obj1(x, y), obj2(x, y)
# def equations(vars):
#     x, y = vars
#     eq1 =  #diff((y-1)*exp(4*(2.75-y))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), y) #lambdify(y, diff((y-1)*exp(4*(2.75-y))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), y), modules=['numpy']) # Example equation 1
#     eq2 = -(x-1)*exp(4*(2-x))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1) #diff((x-1)*exp(4*(2-x))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), x)#lambdify(x, diff((x-1)*exp(4*(2-x))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), x), modules=['numpy'])      # Example equation 2
#     return (eq1, eq2)

# 2. Choose a solver and 3. Provide an initial guess
# initial_guess = np.array([1.5, 1.5])
x0, y0 = 1.5, 1.5
# 4. Call the solver
x_opt, y_opt, obj1_opt, obj2_opt = optimize_iteratively(x0, y0)

print("Optimized x:", x_opt)
print("Optimized y:", y_opt)
print("Optimized objective function 1:", -obj1_opt)
print("Optimized objective function 2:", -obj2_opt)
demand1=exp(4*(a1-x_opt))/(exp(4*(a2-y_opt)) + exp(4*(a1-x_opt)) + 1)
demand2=exp(4*(a2-y_opt))/(exp(4*(a2-y_opt)) + exp(4*(a1-x_opt)) + 1)
print("Optimized demand 1:", demand1)
print("Optimized demand 2:", demand2)

print("Market share:", demand2/(demand2+demand1))
# x, y = symbols('x y')
# eq1 = diff((y-1)*exp(4*(2.75-y))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), y)
# eq2 = diff((x-1)*exp(4*(2-x))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), x)

# func_np = lambdify([y,x], Eq(diff((y-1)*exp(4*(2.75-y))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), y),
#                  diff((x-1)*exp(4*(2-x))/(exp(4*(2.75-y)) + exp(4*(2-x)) + 1), x)), modules=['numpy'])
# solutions = fsolve(func_np, (0.5, 0.5))
# print(solutions) # Output: {x: 1, y: -1}