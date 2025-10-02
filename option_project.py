#To run : streamlit run greek_app_v2.py
import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import opstrat as op
import yfinance as yf
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')  # modern grid + soft colors
class Option:
    #Use of abstraction
    def __init__(self,S,K,T,sig,r):

        self.S = S
        self.K = K
        self.T = T
        self.sig = sig
        self.r = r

    def d1(self):
        return (np.log(self.S/self.K)+(self.r+(self.sig**2)/2)*self.T)/(self.sig*np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sig*np.sqrt(self.T)

    #Constant greek across options:
    def gamma(self):
      return norm.pdf(self.d1())/(self.S*self.sig*np.sqrt(self.T))
    
    def vega(self):
      return self.S*np.sqrt(self.T)*norm.pdf(self.d1())
    

class Call(Option):
    
    def price(self):
      return self.S*norm.cdf(self.d1()) - self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2())

    def delta(self):
      return norm.cdf(self.d1())

    def theta(self):
      return -self.S*self.sig*norm.pdf(self.d1())/(2*np.sqrt(self.T)) - self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2())

    def rho(self):
      return (self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2()))

class Put(Option):
    
    def price(self):
      return self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2()) - self.S*norm.cdf(-self.d1())

    def delta(self):
      return -norm.cdf(-self.d1())

    def theta(self):
      return -self.S*self.sig*norm.pdf(self.d1())/(2*np.sqrt(self.T)) + self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2())

    def rho(self):
      return self.K*self.T*np.exp(-self.r*self.T)*(norm.cdf(self.d2())-1)



class GreekGraph(Option):
  #2d Graphs
  def graph_all_call_greeks_v_spot(self):
    S = np.linspace(0.6*self.S, 1.4*self.S, 100)
    delta=[]
    gamma=[]
    vega=[]
    theta=[]
    rho=[]
    for s in S:
      call = Call(s, self.K, self.T, self.sig, self.r)
      delta.append(call.delta())
      gamma.append(call.gamma())
      vega.append(call.vega())
      theta.append(call.theta())
      rho.append(call.rho())
    fig, axs = plt.subplots(5,1, figsize=(10,18), sharex=False)

    axs[0].plot(S, delta, linewidth=1, color='royalblue')
    axs[0].set_title('Delta of call option', fontsize=16, fontweight='bold')
    axs[0].set_ylabel('Delta')
    axs[0].grid(True)
    axs[0].set_xlabel('Spot Price')

    axs[1].plot(S, gamma, linewidth=1, color='royalblue')
    axs[1].set_title('Gamma of call option', fontsize=16, fontweight='bold')
    axs[1].set_ylabel('Gamma')
    axs[1].grid(True)
    axs[1].set_xlabel('Spot Price')

    axs[2].plot(S, vega, linewidth=1, color='royalblue')
    axs[2].set_title('Vega of call option', fontsize=16, fontweight='bold')
    axs[2].set_ylabel('Vega')
    axs[2].grid(True)
    axs[2].set_xlabel('Spot Price')

    axs[3].plot(S, theta, linewidth=1, color='royalblue')
    axs[3].set_title('Theta of call option', fontsize=16, fontweight='bold')
    axs[3].set_ylabel('Theta')
    axs[3].grid(True)
    axs[3].set_xlabel('Spot Price')

    axs[4].plot(S, rho, linewidth=1, color='royalblue')
    axs[4].set_title('Rho of call option', fontsize=16, fontweight='bold')
    axs[4].set_ylabel('Rho')
    axs[4].grid(True)
    axs[4].set_xlabel('Spot Price')

    for ax in axs:
        ax.axvline(self.K, color='black', linestyle='--', linewidth=1, label='ATM (S=K)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig
  def graph_all_put_greeks_v_spot(self):
      S = np.linspace(0.6*self.S, 1.4*self.S, 100)
      delta=[]
      gamma=[]
      vega=[]
      theta=[]
      rho=[]
      for s in S:
        put = Put(s, self.K, self.T, self.sig, self.r)
        delta.append(put.delta())
        gamma.append(put.gamma())
        vega.append(put.vega())
        theta.append(put.theta())
        rho.append(put.rho())
      fig, axs = plt.subplots(5,1, figsize=(10,18), sharex=False)

      axs[0].plot(S, delta, linewidth=1, color='royalblue')
      axs[0].set_title('Delta of Put option', fontsize=16, fontweight='bold')
      axs[0].set_ylabel('Delta')
      axs[0].grid(True)
      axs[0].set_xlabel('Spot Price')

      axs[1].plot(S, gamma, linewidth=1, color='royalblue')
      axs[1].set_title('Gamma of Put option', fontsize=16, fontweight='bold')
      axs[1].set_ylabel('Gamma')
      axs[1].grid(True)
      axs[1].set_xlabel('Spot Price')

      axs[2].plot(S, vega, linewidth=1, color='royalblue')
      axs[2].set_title('Vega of Put option', fontsize=16, fontweight='bold')
      axs[2].set_ylabel('Vega')
      axs[2].grid(True)
      axs[2].set_xlabel('Spot Price')

      axs[3].plot(S, theta, linewidth=1, color='royalblue')
      axs[3].set_title('Theta of Put option', fontsize=16, fontweight='bold')
      axs[3].set_ylabel('Theta')
      axs[3].grid(True)
      axs[3].set_xlabel('Spot Price')

      axs[4].plot(S, rho, linewidth=1, color='royalblue')
      axs[4].set_title('Rho of Put option', fontsize=16, fontweight='bold')
      axs[4].set_ylabel('Rho')
      axs[4].grid(True)
      axs[4].set_xlabel('Spot Price')

      for ax in axs:
          ax.axvline(self.K, color='black', linestyle='--', linewidth=1, label='ATM (S=K)')

      plt.tight_layout(rect=[0, 0, 1, 0.96])
      plt.show()
      return fig

  #Surface Graphs
  def graph_gamma_surface(self):
    S = np.linspace(0.7*self.S, 1.3*self.S, 50)
    T = np.linspace(1e-8, self.T, 50)
    S,T = np.meshgrid(S,T)
    gamma= np.zeros_like(S)
    for s in range(S.shape[0]):
      for t in range(T.shape[0]):
        s_val = S[s,t]
        t_val = T[s,t]
        opt = Option(s_val, self.K, t_val, self.sig, self.r)
        gamma[s,t]= opt.gamma()
    fig = go.Figure(data=[go.Surface(z=gamma, x=S, y=T, colorscale='Viridis')])
    fig.update_layout(title='Gamma Surface', autosize=True, scene=dict(
        xaxis_title = 'Spot Price',
        yaxis_title = 'Time to Maturity',
        zaxis_title = 'Gamma'
    ))
    return fig

  def graph_vega_surface(self):
      S = np.linspace(0.7*self.S, 1.3*self.S, 50)
      T = np.linspace(1e-8, self.T, 50)
      S,T = np.meshgrid(S,T)
      vega= np.zeros_like(S)
      for s in range(S.shape[0]):
        for t in range(T.shape[0]):
          s_val = S[s,t]
          t_val = T[s,t]
          opt= Option(s_val, self.K, t_val, self.sig, self.r)
          vega[s,t] = opt.vega()
      fig = go.Figure(data=[go.Surface(z=vega, x=S, y=T)])
      fig.update_layout(title='Vega Surface', scene=dict(xaxis_title='Spot',yaxis_title='Time To Maturity', zaxis_title='Vega'))
      return fig

class OptionLeg:
   def __init__(self,option_type, position, S, K, T, r, sig, choice_mat, quantity=1):
      """
      option_type = 'call' or 'put',
      position = 'Long' or 'Short',
      """
      self.option_type = option_type.lower()
      self.position = position.lower()
      self.quantity = quantity
      self.choice_mat = choice_mat
      if self.option_type == 'call':
         self.option = Call(S,K,T,sig,r)
      elif self.option_type == 'put':
         self.option = Put(S,K,T,sig,r) 
      else:
         raise ValueError('Option Type must be call or put')

   def __repr__(self):
      sign = '+' if self.position == "long" else '-'

      return f'{sign}{self.quantity}{self.option_type.upper()}, K={self.option.K}, T={self.choice_mat}' 
   

class Strategy:
   def __init__(self, underlying):
      self.underlying=underlying
      self.legs=[]
  
   def add_leg(self, leg:OptionLeg):
      self.legs.append(leg)
   
   def summary(self):
      return f'Strategy on {self.underlying}\n' + '\n'.join(str(leg) for leg in self.legs)
   
   def value(self):
      total = 0
      for leg in self.legs:
         val = leg.option.price()
         if leg.position == 'short':
            val = -val 
         total+=leg.quantity * val
      return total
   
   def greeks(self):
      totals = {'delta' : 0, 'gamma' : 0, 'vega' : 0, 'theta' : 0, 'rho' : 0}
      for leg in self.legs:
         mult = 1 if leg.position == 'long' else -1
         totals['delta'] += mult * leg.quantity * leg.option.delta()
         totals['gamma'] += mult * leg.quantity * leg.option.gamma()
         totals['vega'] += mult * leg.quantity * leg.option.vega()
         totals['theta'] +=  mult * leg.quantity *leg.option.theta()
         totals['rho'] +=  mult * leg.quantity * leg.option.rho()
      return totals

class StrategyGreekGraph:
   def __init__(self, strategy):
      self.strategy = strategy 
  
   def graph_all_greeks_v_spot(self):
      stock = yf.Ticker(self.strategy.underlying)
      S_0 = stock.fast_info['lastPrice']
      S = np.linspace(0.6*S_0, 1.4*S_0)
      delta = []
      gamma = []
      vega = []
      theta = []
      rho = []
      for s in S:
         delta_val = 0
         gamma_val = 0
         vega_val = 0
         theta_val = 0
         rho_val = 0

         #Delta part:
         for leg in self.strategy.legs:
            if leg.option_type == 'call':
               opt=Call(s, leg.option.K, leg.option.T, leg.option.sig, leg.option.r)
            else:
               opt=Put(s, leg.option.K, leg.option.T, leg.option.sig, leg.option.r)
            mult = -1 if leg.position== 'short' else 1
            
            delta_val += mult * leg.quantity * opt.delta()
            gamma_val += mult * leg.quantity * opt.gamma()
            vega_val += mult * leg.quantity * opt.vega()
            theta_val += mult * leg.quantity * opt.theta()
            rho_val += mult * leg.quantity * opt.rho()
         delta.append(delta_val)
         gamma.append(gamma_val)
         vega.append(vega_val)
         theta.append(theta_val)
         rho.append(rho_val)

      fig, axs = plt.subplots(5,1, figsize=(10,18), sharex=False)

      axs[0].plot(S, delta, linewidth=1, color='royalblue')
      axs[0].set_title('Delta of Put option', fontsize=16, fontweight='bold')
      axs[0].set_ylabel('Delta')
      axs[0].grid(True)
      axs[0].set_xlabel('Spot Price')

      axs[1].plot(S, gamma, linewidth=1, color='royalblue')
      axs[1].set_title('Gamma of Put option', fontsize=16, fontweight='bold')
      axs[1].set_ylabel('Gamma')
      axs[1].grid(True)
      axs[1].set_xlabel('Spot Price')

      axs[2].plot(S, vega, linewidth=1, color='royalblue')
      axs[2].set_title('Vega of Put option', fontsize=16, fontweight='bold')
      axs[2].set_ylabel('Vega')
      axs[2].grid(True)
      axs[2].set_xlabel('Spot Price')

      axs[3].plot(S, theta, linewidth=1, color='royalblue')
      axs[3].set_title('Theta of Put option', fontsize=16, fontweight='bold')
      axs[3].set_ylabel('Theta')
      axs[3].grid(True)
      axs[3].set_xlabel('Spot Price')

      axs[4].plot(S, rho, linewidth=1, color='royalblue')
      axs[4].set_title('Rho of Put option', fontsize=16, fontweight='bold')
      axs[4].set_ylabel('Rho')
      axs[4].grid(True)
      axs[4].set_xlabel('Spot Price')

      for ax in axs:
          ax.axvline(S_0, color='black', linestyle='--', linewidth=1, label='ATM (S=K)')

      plt.tight_layout(rect=[0, 0, 1, 0.96])
      plt.show()
      return fig
         


# VOLATILITY FUNCTIONS
#x_n+1 = x_n - f(x_n)/f'(x_n) 
def newton_raphson_implied_vol(S,K,T,r,sig, opt_type, market_price, max_iter, method='black_scholes'):
   tol=1e-3
   if method=='black_scholes':
      
      for i in range(max_iter):
        if opt_type == 'Call':
          theo = Call(S,K,T,sig,r).price()
        else:
           theo = Put(S,K,T,sig,r).price()

        vega = Option(S,K,T,sig,r).vega()
        
        diff = theo-market_price
        sig_next = sig - diff/vega

        if abs(sig_next - sig)<tol:
           return sig_next
        
        sig = sig_next
               
      return sig_next
      
def brentq(f, a, b, tol=1e-3, max_iter=1000):
  fa, fb = f(a), f(b)

  if fa*fb > 0:
    raise ValueError(f'Root is not bracketed in [{a},{b}]')
  if abs(fa) < abs(fb):
    a,b = b,a 
    fa,fb = fb,fa

  c, fc = a, fa 
  d = e = b-a

  for _ in range(max_iter):
     if abs(fb)<tol:
      return b
     
     if fa != fc and fb != fc: #inverse quad interpolation if fa=!fb!=fc (because need 3 data points)
        s = ((a*fb*fc) / ((fa-fb) * (fa-fc))  
        + (b*fa*fc) / ((fb-fa) * (fb-fc))  
        + (c*fa*fb) / ((fc-fa) * (fc-fb)))
     else: #secant method with 2 points
        s = b - fb * (b-a)/(fb-fa)
      
     #conditions for interpolation, else bisection
     cond1 = not ((3*a+b)/4 < s < b) if a<b else not(b<s<(3*a+b)/4)
     cond2 = (abs(s-b) >= abs(b-c)/2)
     cond3 = (abs(b-c) < tol)
     cond4 = (abs(fb) > abs(fc))

     if cond1 or cond2 or cond3 or cond4:
        s = 0.5*(a+b)
        d = e = b-a
     fs = f(s)
     c, fc = b, fb

     if fa*fs<0:
        b,fb = s, fs
     else:
        a, fa= s,fs

     if abs(fa) < abs(fb):
        a,b = b,a
        fa, fb = fb, fa

     if abs(b-a) < tol: 
        return b
     
def implied_vol_brent(S,K,T,r, opt_type, market_price):
   def f(sig):
      if opt_type=='Call':
         return Call(S,K,T,sig,r).price() - market_price
      else:
         return Put(S,K,T,sig,r).price() - market_price
   try: 
      return brentq(f, 1e-2, 5.0, tol = 1e-3, max_iter=1000)
   except ValueError:
      return np.nan
  
#Final implied vol function 
def implied_vol(S, K, T, r, guess, opt_type, market_price):
    sig = newton_raphson_implied_vol(S, K, T, r, guess, opt_type, market_price, 1000)
    if np.isnan(sig):
        sig = implied_vol_brent(S, K, T, r, opt_type, market_price)
    return sig

#Vol surface
def vol_surface(S, symbol, option_type, r):
        stock = yf.Ticker(symbol)
        maturities = stock.options
        surface_data = []
        for expiry in maturities:
           expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
           T = (expiry_date - datetime.today()).days/252

           option_chain = stock.option_chain(expiry)
           chain= option_chain.calls if option_type=='Call' else option_chain.puts

           chain = chain[(chain['strike'] >= 0.7 * S) &
                      (chain['strike'] <= 1.3 * S)]
           last_iv = 0.2
           for _, row in chain.iterrows():
              K = row['strike']
              market_price = row['lastPrice']
              iv = implied_vol(S,K,T,r,last_iv, option_type, market_price)
              last_iv = iv
              surface_data.append([K,T,iv])
        df = pd.DataFrame(surface_data)
        return df
      

st.title('Option Pricer and Greeks Visualizer')
underlyings = {
   'Apple' : 'AAPL',
   "Google" : "GOOG", 
   "Microsoft" : "MSFT",
   "Amazon" : "AMZN",
   "Nvidia" : "NVDA"
}



#Option type
st.title('Option Pricer')
option_ss, option_strategy, structured_products = st.tabs(['Single Option', 'Option Strategy', 'Structured Products'])

with option_ss:
  option_type = st.selectbox('Choose an option type', ['Call', 'Put'])

  underlying_choice = st.selectbox('Choose an Underlying', list(underlyings.keys()))
  symbol = underlyings[underlying_choice]
  stock = yf.Ticker(symbol)
  S = stock.fast_info['lastPrice']
  st.write(f'Current price is {round(S,2)}')
  
  maturities = list(stock.options)
  choice_mat = st.selectbox('Choose a maturity', maturities)


  #Strike param
  option_chain_all = stock.option_chain(choice_mat)
  option_chain = option_chain_all.calls if option_type == 'Call' else option_chain_all.puts
  strikes = list(option_chain['strike'])
  strike_choice = st.selectbox('Choose a strike', strikes, index=int(len(strikes)*0.6))
  K = strike_choice

  #Time Parameter
  today = datetime.today()
  expiry = datetime.strptime(choice_mat, '%Y-%m-%d')
  T = (expiry-today).days/252

  #Interest rate parameter
  r = yf.Ticker('^IRX').fast_info['lastPrice']/100

  #Vol calculation : newton raphson method
  market_price = (option_chain[option_chain['strike'] == K]['ask']).item()
  guess=0.2
  sig = implied_vol(S,K,T,r,guess,option_type, market_price)

  option_obj = Call(S,K,T,sig,r) if option_type == 'Call' else Put(S,K,T,sig,r)
  price = option_obj.price()
  delta = option_obj.delta()
  gamma = option_obj.gamma()
  theta = option_obj.theta()
  vega = option_obj.vega()
  rho = option_obj.rho()



  #On button click
  if st.button('Show Analytics'):
      data = {
         'Underlying' : [symbol],
         'Current Underlying Price': [round(S,2)],
         'Option Type' : [option_type],
         'Option Price' : [round(price,2)],
         'Market Price' : market_price,
         'Strike' : [K],
         'Time to maturity' : [round(T,2)],
         'Implied Volatility' : [round(sig,2)],
         'Delta' : [round(delta,2)],
         'Gamma' : [round(gamma,3)],
         'Vega' : [round(vega,3)],
         'Theta' : [round(theta,3)], 
         'Rho' : [round(rho,2)]
      }
      option_analytics_df = pd.DataFrame(data, index=None).T
      option_analytics_df.columns = ['Value']
      st.table(option_analytics_df)
      gg = GreekGraph(S,K,T,sig,r)
      #gg.S, gg.K, gg.T, gg.sig, gg.r = S,K,T,sig,r
      fig = gg.graph_all_call_greeks_v_spot() if option_type == 'Call' else gg.graph_all_put_greeks_v_spot()
          
      st.pyplot(fig)
      
      fig_gamma = gg.graph_gamma_surface()
      st.plotly_chart(fig_gamma)

      fig_vega = gg.graph_vega_surface()
      st.plotly_chart(fig_vega)

      #Volatility surface plot
      vol_surface_df = vol_surface(S, symbol, option_type, r)
      fig_surf_vol = go.Figure(data = [go.Surface(
         x=vol_surface_df['K'],
         y=vol_surface_df['T'],
         z=vol_surface_df['iv'],
         colorscale='viridis'
      )])
      fig_surf_vol.update_layout(title='Implied Volatility Surface', scene=dict(xaxis_title='Strike', yaxis_title='Time to maturity', zaxis_title='Implied Volatility'))
      st.plotly_chart(fig_surf_vol)

strategies = [
   'Custom',
   'Covered Call', #Long UL, Short 1 OTM call
   'Protective Put', #Long UL, Long 1 OTM put
   'Call Spread', #Long 1 Call ITM, short 1 call OTM
   'Put Spread', #Long 1 Put ITM, short 1 call OTM
   'Box Spread', #Long 1 Call Spread (K1,K2), Long 1 Put Spread (same K1,K2)
   'Butterfly Spread (Calls)', #Long 1 Call K1, Short 2 Call K2, Long 1 Call K3
   'Butterfly Spread (Puts)', #Long 1 Put K1, Short 2 Put K2, Long 1 Put K3
   'Calendar Spread (Calls)', #Short 1 Call K with T1, Long 1 Call K with T2
   'Calendar Spread (Puts)', #Short 1 Put K with T1, Long 1 Call K with T2
   'Straddle', #Long 1 call and 1 put with same K
   'Strangle', #Long 1 put K1 and 1 call K2, K1<K2
   'Strip', #Long 1 call, 2 puts with same K
   'Strap' #Long 2 calls, 1 put with same K
]
with option_strategy:
   strat = st.selectbox('Choose a strategy', strategies)
   if strat == 'Custom':
      n_legs = st.number_input('Number of Legs', min_value=1, max_value=10, value=2)
      underlying_choice = st.selectbox(f'Select an Underlying', list(underlyings.keys())) 
      symbol = underlyings[underlying_choice]
      stock = yf.Ticker(symbol)
      S = stock.fast_info['lastPrice']
      st.write(f'Current price is {round(S,2)}')
      strategy = Strategy(symbol)
      op_list = []
      for i in range(n_legs):
         st.subheader(f'Option leg {i+1}')
         option_type = st.selectbox(f'Select option type (Leg {i+1})', ['Call', 'Put'])
         position = st.selectbox(f'Choose position for leg {i+1}', ['Long', 'Short'])
         quantity = st.number_input(f'Choose the quantity for leg {i+1}', min_value=1, max_value=10, value=1)
         
          
         maturities = list(stock.options)
         choice_mat = st.selectbox(f'Select a maturity (Leg {i+1})', maturities)


          #Strike param
         option_chain_all = stock.option_chain(choice_mat)
         option_chain = option_chain_all.calls if option_type == 'Call' else option_chain_all.puts
         strikes = list(option_chain['strike'])
         strike_choice = st.selectbox(f'Select a strike (Leg {i+1})', strikes, index=int(len(strikes)*0.6))
         K = strike_choice

          #Time Parameter
         today = datetime.today()
         expiry = datetime.strptime(choice_mat, '%Y-%m-%d')
         T = (expiry-today).days/252

          #Interest rate parameter
         r = yf.Ticker('^IRX').fast_info['lastPrice']/100

          #Vol calculation : newton raphson method
         market_price = (option_chain[option_chain['strike'] == K]['ask']).item()
         guess=0.2
         sig = implied_vol(S,K,T,r,guess,option_type, market_price)

         option_obj = Call(S,K,T,sig,r) if option_type == 'Call' else Put(S,K,T,sig,r)
         mult = 1 if position == 'Long' else -1
         #price += mult * quantity * option_obj.price()
         #delta += mult * quantity * option_obj.delta()
         #gamma += mult * quantity * option_obj.gamma()
         #theta += mult * quantity * option_obj.theta()
         #vega  += mult * quantity * option_obj.vega()
         #ho   += mult * quantity * option_obj.rho()
         option = OptionLeg(option_type, position, S,K,T,r,sig, choice_mat, quantity)
         strategy.add_leg(option)
         opstrat_type = 'c' if option_type =='Call' else 'p'
         opstrat_pos = 'b' if position == 'Long' else 's'
         opstrat_price = option_obj.price()
         op_list.append({'op_type':opstrat_type,'strike':K,'tr_type':opstrat_pos,'op_pr':opstrat_price})
      if st.button('Show Strategy Analytics'):
         price = strategy.value()
         greeks= strategy.greeks()
         data_strategy = {
         'Underlying' : [symbol],
         'Components' : strategy.summary(),
         'Option Price' : [round(price,2)],
         'Current Underlying Price': [round(S,2)],
         'Strike' : [K],
         'Time to maturity' : [round(T,2)],
         'Implied Volatility' : [round(sig,2)],
         'Delta' : [round(greeks['delta'],2)],
        'Gamma' : [round(greeks['gamma'],3)],
        'Vega'  : [round(greeks['vega'],3)],
        'Theta' : [round(greeks['theta'],3)],
        'Rho'   : [round(greeks['rho'],2)]
      }
         df_analytics_strategy = pd.DataFrame(data_strategy).T
         df_analytics_strategy.columns = ['Value']
        #Results
         st.table(df_analytics_strategy)

        #Payoff graph
         op.multi_plotter(spot=S, spot_range=10, op_list=op_list)
         payoff_plot = plt.gcf()
         st.pyplot(payoff_plot)
        #Graphs
         greek_graph_strategy = StrategyGreekGraph(strategy)
         fig = greek_graph_strategy.graph_all_greeks_v_spot()
         st.pyplot(fig)
         

#ADD STRATEGY TO ADD THE LEGS FROM THE LOOP
          