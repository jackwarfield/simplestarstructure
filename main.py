### import statements ###
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from numba import njit,jit

### define constants ###
mh = 1.6733e-24 # grams
kb = 1.380658e-16 # erg/K
arad = 7.5646e-15 # erg/cc/K4
Msun = 1.99e33 # g
G = 6.67259e-8 # cc/g/s
c = 2.99792458e10 # cm/s
Rsun = 6.957e10

### other functions ###
@njit()
def RK4(
    r: float,
    y: Iterable[float],
    f: Callable,
    params: Iterable,
    h: float = 0.001,
    ) -> Iterable[float]:
  hh = 0.5*h
  h6 = h/6.0
  rh = r+hh

  dydx = f(r,y,params)
  yt = y+hh*dydx
  dyt = f(rh,yt,params)
  yt = y+dyt*hh
  dym = f(rh,yt,params)
  yt = y+dym*h
  dym = dyt+dym
  r += h
  dyt = f(r,yt,params)

  y += h6*(dydx + dyt + 2*dym)

  return y

@njit()
def get_rho(
    p: float,
    t: float,
    mu: float,
    ) -> float:
  coeff = mu/(kb*t)
  term2 = arad*t**4/3
  return coeff*(p-term2)

@njit()
def specent(
    T: float,
    P: float,
    mu: float,
    ) -> float:
  rho = get_rho(p,t,mu)
  first = 3*kb/(2*m)*np.log(T*rho**(-2/3))
  sec = 4*a*T**3/(3*rho)
  return first+sec

@njit()
def dpdr(
    r: float,
    m: float,
    rho: float,
    ) -> float:
  return -rho*G*m/r**2

@njit()
def dmdr(
    r: float,
    rho: float,
    ) -> float:
  return 4*np.pi*r*r*rho

@njit()
def epsilon(
    t: float,
    rho: float,
    X: float,
    ) -> float:
  Xcn = 0.01
  tau_pp = 33.80*(t/1e6)**(-1/3)
  tau_cn = 152.3*(t/1e6)**(-1/3)
  term1 = 2.12e3*X*tau_pp**2*np.exp(-tau_pp)
  term2 = 3.70e23*Xcn*tau_cn**2*np.exp(-tau_cn)
  return X*rho*(term1+term2)

@njit()
def dldr(
    r: float,
    t: float,
    rho: float,
    X: float,
    ) -> float:
  eps = epsilon(t,rho,X)
  return 4*np.pi*r*r*rho*eps

@njit()
def kappa(
    t: float,
    rho: float,
    X: float,
    Z: float,
    ) -> float:
  term1 = 3e25*Z*(1+X)*rho*t**(-3.5)
  term2 = 0.2*(1+X)
  return term1+term2

@njit()
def del_ad(
    t: float,
    mu: float,
    rho: float,
    ):
  theta = arad*t**3/(rho/mu*kb)
  top = 18+30*theta+8*theta*theta
  bot = 45+120*theta+32*theta*theta
  return top/bot

@njit()
def del_rad(
    p: float,
    t: float,
    m: float,
    rho: float,
    l: float,
    X: float,
    Z: float,
    ) -> float:
  opac = kappa(t,rho,X,Z)
  top = 3*p*opac*l
  bot = 16*np.pi*G*m*c*arad*t**4
  return top/bot

@njit()
def dtdr(
    r: float,
    p: float,
    t: float,
    m: float,
    rho: float,
    l: float,
    mu: float,
    X: float,
    Z: float,
    ) -> float:
  coeff = -G*m*rho*t/(r*r*p)
  dAD = del_ad(t,mu,rho)
  dRAD = del_rad(p,t,m,rho,l,X,Z)
  if dAD < dRAD:
    return coeff*dAD
  else:
    return coeff*dRAD


@njit()
def snapshot(
    r: float,
    yint: Iterable[float],
    params: Iterable,
    ) -> Iterable[float]:
  p,t,m,l = yint
  mu,X,Z = params
  rho = get_rho(p,t,mu)
  y = np.array([
      dpdr(r,m,rho),
      dtdr(r,p,t,m,rho,l,mu,X,Z),
      dmdr(r,rho),
      dldr(r,t,rho,X),
      ])
  return y

@njit()
def shootloop(
    r0: float,
    rlim: float,
    yint: Iterable[float],
    params: Iterable[float],
    step: float,
    ):
  rout = [r0]
  pout = [yint[1]]
  r = r0
  while r < rlim:
    yint = RK4(r,yint,snapshot,params,h=step)
    r += step
    rout += [r]
    pout += [yint[1]]
  return r,yint,rout,pout

@njit()
def shoot(
    p0: float,
    t0: float,
    mu: float,
    X: float,
    Z: float,
    mstar: float,
    step: float,
    rlim: float = 348000000000.0,
    tstep: Optional[Iterable[float]] = np.array([2e6,1e6]),
    tnum: int = 10000,
    pstep: Optional[Iterable[float]] = None,
    pnum: int = 10000,
    tol: float = 1e-1,
    out: bool = False
    ) -> Iterable[float]:

  params = [mu,X,Z]
  m0,l0 = 0.000001,0.0000001

  ystart = [p0,t0,m0,l0,np.nan,np.nan,np.nan,np.nan,np.nan]
  yresults = [ystart]

  p0a,t0a,m0a,l0a,rr,pr,tr,mr,lr = [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
      [0.0],[0.0]

  #                     rf,    pf,    tf,    mf,    lf
  yint = np.array([p0,t0,m0,l0])

  pf = 1e8
  tf = 1e8

  rstart = step/10
  rout = [rstart]
  pout = [p0]
  tout = [t0]
  mout = [m0]
  lout = [l0]

  tl,pl = t0,p0
  if tstep == None:
    trange = np.array([t0])
  else:
    trange = np.linspace(t0-tstep[0], t0+tstep[1], tnum)
  #trange = [t0]
  if pstep == None:
    prange = np.array([p0])
  else:
    prange = np.linspace(p0-pstep[0], p0+pstep[1], pnum)

  for t0 in trange:
    for p0 in prange:
      r = 1
      yint = np.array([p0, t0, m0, l0])
      while r < rlim:
        tpre = yint[1]
        yint = RK4(r,yint,snapshot,params,h=step)
        r += step
        if out:
          rout += [r]
          pout += [yint[0]]
          tout += [yint[1]]
          mout += [yint[2]]
          lout += [yint[3]]
        #if not (yint[1] > 0):
        #  break
        #if (abs(yint[1]-tpre) < tol):
        #  break
      #if not (yint[1] > 0):
      #  t0 = tl + 2*tstep*(np.random.rand()-0.5)

      p0a += [p0]
      t0a += [t0]
      m0a += [m0]
      l0a += [l0]
      rr += [r]
      pr += [yint[0]]
      tr += [yint[1]]
      mr += [yint[2]]
      lr += [yint[3]]

  return p0a[1:],t0a[1:],m0a[1:],l0a[1:],rr[1:],pr[1:],tr[1:],mr[1:],lr[1:],\
      rout,pout,tout,mout,lout



### main function ###
def main() -> int:
  config = pd.read_json("config.json")

  X = config.fixed.X
  Y = config.fixed.Y
  Z = config.fixed.Z
  mu = 1/((1/mh)*(2*X + 3*Y/4 + Z/2))

  Mstar = config.init.Mstar*Msun
  p0 = config.init.Pc0
  t0 = config.init.Tc0
  tstep = config.init.tstep
  try:
    tstep = eval(tstep)
  except:
    pass
  if type(tstep) == list:
    tstep = np.array(tstep)
  pstep = config.init.pstep
  try:
    pstep = eval(pstep)
  except:
    pass
  if type(pstep) == list:
    pstep = np.array(pstep)
  outtab = eval(config.other.outtable)

  pnum = int(config.init.pnum)
  tnum = int(config.init.pnum)
  
  print("init")
  p0a,t0a,m0a,l0a,rr,pr,tr,mr,lr,rout,pout,tout,mout,lout = \
      shoot(p0,t0,mu,X,Z,Mstar,1e4,rlim=1,tstep=tstep,pstep=pstep,tnum=1,
          pnum=1, out=False
          )
  print("run")
  #p0a,t0a,m0a,l0a,rr,pr,tr,mr,lr,rout,pout = shoot(p0,t0,mu,X,Z,Mstar,1e4)
  p0a,t0a,m0a,l0a,rr,pr,tr,mr,lr,rout,pout,tout,mout,lout = \
      shoot(p0,t0,mu,X,Z,Mstar,step=Rsun/1000, rlim=10*Rsun,
          tstep=tstep, pstep=pstep, out=outtab, pnum=pnum, tnum=tnum
          )
  print("done run")

  with open("results.txt","w") as f:
    print("p0a,t0a,m0a,l0a,rr,pr,tr,mr,lr", file=f)
    for i in range(len(p0a)):
      print(f"{p0a[i]},{t0a[i]},{m0a[i]},{l0a[i]},{rr[i]},{pr[i]},{tr[i]},{mr[i]/Msun},{lr[i]/(3.9e33)}", file=f)

  if outtab:
    with open("structure.txt","w") as f:
      print("r,p,t,m,l", file=f)
      for r,p,t,m,l in zip(rout,pout,tout,mout,lout):
        print(f"{r},{p},{t},{m},{l}", file=f)

  return 0

###############################################################################
###############################################################################

if __name__ == "__main__":
  raise SystemExit(main())
