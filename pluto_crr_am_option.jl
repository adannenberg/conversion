### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ a2901126-3424-11eb-3e40-c76e36480a1a
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ b580be6e-3424-11eb-0b25-718fabf3b6e4
begin
	Pkg.add(["Plots","Optim", "Dates", "Statistics", "CSV", "DataFrames", "Distributions", "PlutoUI","Plotly"])
	using Plots, Optim, Dates, Statistics, CSV, DataFrames, Distributions, PlutoUI
	plotly()
end

# ╔═╡ 14726f12-4d5e-11eb-3f82-575c04cb690a
html"""<style>
main {
    max-width: 1900px;
}
"""

# ╔═╡ cb315d1c-2da1-11eb-232b-3fcccc40e5e6
begin
	m = convert(Matrix, CSV.read("/home/alex/Dropbox/Code/R/ymd.txt",DataFrame,header=false));
	us_trading_holidays = [Date(m[i,1],m[i,2],m[i,3]) for i=1:size(m,1)];
end

# ╔═╡ 983d5ef6-2d9c-11eb-260a-fbc85d734f14
function bd(d, holidays = us_trading_holidays)
	if dayofweek(d) == 6
		d = d+Day(2)
	end
	if dayofweek(d) == 7
		d = d+Day(1)
	end
	while true
		if in(d, holidays)
			d = d+Day(1)
		else
			break
		end
	end
	d
end

# ╔═╡ a4deccaa-449c-11eb-3342-eb6908001a34
function Rmm2Rcc(rmm)
	rcc = 12 * log(1 + rmm * (365/12) / 360)
	rcc
end

# ╔═╡ eb7f9626-449c-11eb-2b83-3f180644b7d1
function Rcc2Rmm(rcc)
	rmm = 360 * 12 / 365 * (exp(rcc/12) - 1)
	rmm
end

# ╔═╡ 72a4d668-2d9e-11eb-26cf-d99bc088183f
function workday(d, n, holidays = us_trading_holidays)
	d = bd(d)
	if n>0
		for i=1:n
			d=bd(d+Day(1))
		end
	end
	d
end

# ╔═╡ 72177212-2da0-11eb-153e-4b8bf66c697c
function t_addnl(D1,D2,N,i,offset)
	tstep = (D2-D1).value/N
	d1 = bd(D1 + Day(ceil(i*tstep)))
	d2 = workday(d1, offset)
	(d2-d1).value
end

# ╔═╡ 6ebf8d5e-2d97-11eb-204a-511e7aa6b14c
function bintree(putcall, S, K, r, vol, dt_exp, N = 500, dt_exdiv = today() + Day(1000), dt_divpmt = today() + Day(1000), divamt = .0000001, dt_trade = bd(today()))
	r = Rmm2Rcc(r)
	premium_settlement = 1
 	exercise_settlement = 2 #biz days
 	# adjust dates to good business day (following)
  	dt_trade = bd(dt_trade)
	dt_settle = workday(dt_trade, premium_settlement)
	dt_exp = bd(dt_exp)
	dt_exdiv = bd(dt_exdiv)
	dt_divpmt = bd(dt_divpmt)
	# now compute time differences (in days)
	t_exp = (dt_exp - dt_trade).value
	t_exdiv = (dt_exdiv - dt_trade).value
	t_divpmt = (dt_divpmt - dt_trade).value
	if t_exp < t_exdiv
		divamt = 0
	end
	tstep = t_exp / N
	i_exdiv = t_exdiv / tstep
 	i_divpmt = t_divpmt / tstep
	R = exp(r * tstep / 365)
   	S0 = S - divamt / R^i_exdiv
	vol = vol * S / S0
	U = exp(vol * sqrt(tstep / 365))
	D = 1 / U
	p = (R - D) / (U - D)
	q = (U - R) / (U - D)
	if lowercase(putcall) == "put"
    	mult = -1.
	else
    	mult = 1.
	end
	# value at maturity, allocates buffer array
	A = [max(0, mult * (S0 * U^i * D^(N-i) - K)) for i = 0:N] * exp(-r * t_addnl(dt_trade, dt_exp, N, N, exercise_settlement) / 365)
	E = copy(A)
	T = N .* ones(N+1)
	
	for t = (N - 1):-1:0
    	divfv = (t < i_exdiv) * divamt / R^(i_divpmt - t)
    	deflator = exp(-r * t_addnl(dt_trade, dt_exp, N, t, exercise_settlement)/365)
    	for i = 0:t
      		x = mult * (S0 * U^i * D^(t - i) + divfv - K) * deflator# early ex value
			y = (q * A[i + 1] + p * A[i + 2]) / R # future value
			A[i + 1] = max(x, y)
      		E[i + 1] = (q * E[i + 1] + p * E[i + 2]) / R
      		if x > y
        		T[i + 1] = t
      		else
        		T[i + 1] = q * T[i + 1] + p * T[i + 2]
	  		end
		end
	end
	M = exp(r * (dt_settle - dt_trade).value / 365)
  	v = [A[1] * M; E[1] * M; T[1] * tstep]
  	## we transpose v so that it can display in 3 adjacent horizontal excel cells instead of vertically
  	return v
end

# ╔═╡ 9ef5f72c-2dd2-11eb-0d89-53e5346ab60b
function conversion(S, K, r, vol, dt_exp, N = 500, dt_exdiv = today() + Day(1000), dt_divpmt = today() + Day(1000), divamt = .0000001, dt_trade = bd(today()), outputEarlyExInfo = false)
	r = Rmm2Rcc(r)
	premium_settlement = 1
 	exercise_settlement = 2 #biz days
 	# adjust dates to good business day (following)
  	dt_trade = bd(dt_trade)
	dt_settle = workday(dt_trade, premium_settlement)
	dt_exp = bd(dt_exp)
	dt_exdiv = bd(dt_exdiv)
	dt_divpmt = bd(dt_divpmt)
	# now compute time differences (in days)
	t_exp = (dt_exp - dt_trade).value
	t_exdiv = (dt_exdiv - dt_trade).value
	t_divpmt = (dt_divpmt - dt_trade).value
	if t_exp < t_exdiv
		divamt = 0
	end
	tstep = t_exp / N
	i_exdiv = t_exdiv / tstep
 	i_divpmt = t_divpmt / tstep
	R = exp(r * tstep / 365)
   	S0 = S - divamt / R^i_exdiv
	vol = vol * S / S0
	U = exp(vol * sqrt(tstep / 365))
	D = 1 / U
	p = (R - D) / (U - D)
	q = (U - R) / (U - D)

	if outputEarlyExInfo
    	numnode = round(Int,(N+2)*(N+1)/2)
		earlyExInfo = Array{Float64}(undef, numnode, 4)
		# col 1 will be -1(put) or 1(call)
		# col 2 will be price
		# col3 will be date
		# col4 will be otherOptPx
    	earlyExInfoRow = 1
	end
  
		
	# value at maturity, allocates buffer array
	P = [max(0, -1. * (S0 * U^i * D^(N-i) - K)) for i = 0:N] * exp(-r * t_addnl(dt_trade, dt_exp, N, N, exercise_settlement) / 365)
	C = [max(0, +1. * (S0 * U^i * D^(N-i) - K)) for i = 0:N] * exp(-r * t_addnl(dt_trade, dt_exp, N, N, exercise_settlement) / 365)
	
	for t = (N - 1):-1:0
		dt = (bd(dt_trade + Day(ceil(t*tstep)))-dt_trade).value
    	divfv = (t < i_exdiv) * divamt / R^(i_divpmt - t)
    	deflator = exp(-r * t_addnl(dt_trade, dt_exp, N, t, exercise_settlement)/365)
    	for i = 0:t
      		xput = -1. * (S0 * U^i * D^(t - i) + divfv - K) * deflator# early ex value
			yput = (q * P[i + 1] + p * P[i + 2]) / R # future value
			P[i + 1] = max(xput, yput)
			xcall = -xput
			ycall = (q * C[i + 1] + p * C[i + 2]) / R
			C[i + 1] = max(xcall, ycall)
			if outputEarlyExInfo
				if xput >= yput 
				  earlyExInfo[earlyExInfoRow,1] = -1
				  earlyExInfo[earlyExInfoRow,2] = divfv + S0 * U ^ i * D ^ (t - i)
				  earlyExInfo[earlyExInfoRow,3] = dt
				  earlyExInfo[earlyExInfoRow,4] = C[i+1]
				  earlyExInfoRow = earlyExInfoRow + 1
				# call never exercised early if (r >0 and no dividends)
				# call can only be exercised early if t is t_exdiv-1
				elseif (((xcall >= ycall) & (r < 0)) | ((xcall >= ycall) & (r >= 0) & (divamt > 0) & (t+1 == t_exdiv)))
				  earlyExInfo[earlyExInfoRow,1] = 1
				  earlyExInfo[earlyExInfoRow,2] = divfv + S0 * U ^ i * D ^ (t - i)
				  earlyExInfo[earlyExInfoRow,3] = dt
				  earlyExInfo[earlyExInfoRow,4] = P[i+1]
				  earlyExInfoRow = earlyExInfoRow + 1
				end
			end
		end
	end
	if outputEarlyExInfo 
		earlyExInfo = earlyExInfo[1:(earlyExInfoRow-1),:]
		indx = findall(earlyExInfo[:,2] .> 0.01)
		earlyExInfo = earlyExInfo[indx,:]
	end
	M = exp(r * (dt_settle - dt_trade).value / 365)
  	conversion = P[1] * M - C[1]*M + S - K
	if outputEarlyExInfo
		return(C[1]*M, P[1]*M, earlyExInfo)
	else
  		return [conversion, P[1]*M, C[1]*M]
	end
end

# ╔═╡ dae9c4ba-4ca5-11eb-2c3d-75c98eb5d8b9
function mygrapher2(S,
                   K,
                   r_stock,
                   r_option,
                   vol,
                   dt_exp,
                   N = 100,
                   dt_exdiv = today() + Day(1000),
                   dt_divpmt = today() + Day(1000),
                   divamt = 0.000001,
                   dt_trade = bd(today()))
  
	# adjust dates to good business day (following)
  	dt_trade = bd(dt_trade)
  	dt_exp = bd(dt_exp)

  	# now compute time differences (in days)
  	t_exp = (dt_exp - dt_trade).value
  	t_exdiv = (dt_exdiv - dt_trade).value
  	t_divpmt = (dt_divpmt - dt_trade).value
  	
	# for purposes of finding the exercise boundary, we'll use 1 step per calendar day
	# hence we set N = t_exp + 1
  	call, put, obj = conversion(S,K,r_option,vol,dt_exp,5*t_exp+1,dt_exdiv,dt_divpmt,divamt,dt_trade,true)
  	boundary_info = zeros(Float64,t_exp+1,6)# Array{Float64}(undef,t_exp+1,6)
	# column 1 will be days from tradedate
	# column 2 will be call boundary
	# column 3 will be put price at call boundary
	# column 4 will be put boundary
	# column 5 will be call price at put boundary
	# column 6 will be conversion pnl assuming early ex and zero value to unex option
	# column 7 will be upper bound on conversion pnl
	# column 8 will b lower bound on conversion pnl
	boundary_info[:,1] = 0:t_exp
  	for i in 1:t_exp+1
		d = boundary_info[i,1]
		cindx = intersect(findall(obj[:,1] .== 1), findall(obj[:,3] .== d))
		if length(cindx)>0
			boundary_info[i,2] = minimum(obj[cindx,2])
			boundary_info[i,3] = maximum(obj[cindx,4])
		end
		pindx = intersect(findall(obj[:,1] .== -1), findall(obj[:,3] .== d))
		if length(pindx)>0
			boundary_info[i,4] = maximum(obj[pindx,2])
			boundary_info[i,5] = maximum(obj[pindx,4])
		end
	end
	# remove all boundary prices that are less than 1 cent because they're crap
	for c=[2,4]
		indx = findall(boundary_info[:,c] .< .01)
		if length(indx) > 0
			boundary_info[indx,c] .= NaN
		end
	end
	# round all prices to nearest penny
	boundary_info[:,2:5] = round.(boundary_info[:,2:5],digits=2)
	t = 0:t_exp
	
	p1 = scatter(boundary_info[:,1],boundary_info[:,4],xlims = (0,t_exp),label="put exercise bndry")
	scatter!(p1,boundary_info[:,1], boundary_info[:,2],label = "call exercise bndry")
  	
    r_option = Rmm2Rcc(r_option)
    r_stock = Rmm2Rcc(r_stock)
    pnl = (call-put) * exp.(r_option * t / 365) - S * exp.(r_stock * t / 365) .+ K
    tindx = findall(t .>= t_exdiv)
	if length(tindx)>0
		pnl[tindx] = pnl[tindx] + divamt * exp.(-r_option * (t_divpmt .- t[tindx]) / 365)
	end
	boundary_info[:,6] = pnl
	# if market tanks, put will be exercised at put boundary and need to buy back call, reducing pnl by whatever its price is.  therefore this defines a lower bound ( note that prices lower than put boundary will lead to smaller call prices that reduce pnl less)
	# if market spikes, call may be exercised at call boundary and we get sell put, increasing pnl by whatever its price is.  therefore this defines an upper bound (note that prices above the call boundary will lead to lower put prices that increase our pnl less)
	pnl_upper_bound = pnl + boundary_info[:,3]
	pnl_lower_bound = pnl - boundary_info[:,5]
    p2 = plot(boundary_info[:,1],pnl_upper_bound,label="pnl upper bound")
	plot!(p2,boundary_info[:,1],pnl_lower_bound,label="pnl lower bound")
	p3 = scatter(t,boundary_info[:,3],label = "put price at call bndry")
	p4 = scatter(t,boundary_info[:,5],label = "call price at put bndry")
	plot(p1,p2,p3,p4,layout=(2,2),legend=false,size=(800,600))
end


# ╔═╡ 4ba87eea-4cad-11eb-1d51-ad3b81b92cfa
mygrapher2(100,100,-.01,-.01,.12,today()+Day(90),90,today()+Day(30),today()+Day(40),10,today())

# ╔═╡ e37960e8-4d38-11eb-048c-d5df65446524
let	dt_trade = today()
	dt_exp = today() + Day(10)
	N = (dt_exp - dt_trade).value
	dt_trade + Day.(0:N)
end

# ╔═╡ 3394a1c6-2e0a-11eb-0847-6f070b42fc67
function implied_funding_american(conversion_px, 
                   S,
                   K,
                   vol,
                   dt_exp,
                   N = 400,
                   dts_exdiv = today() + Day(1000),
                   dts_divpmt = today() + Day(1000),
                   divamts = 0.000001,
                   dt_trade = bd(today()))
	obj(x) = (conversion(S,K,x,vol,dt_exp,N,dts_exdiv,dts_divpmt,divamts,dt_trade)[1,1] - conversion_px)^2
	res = optimize(obj, -10, 10)
	r_mmm = Optim.minimizer(res)
	r_mmm
end

# ╔═╡ d6747d5a-2eb2-11eb-1e13-670a197e8b1f
function option_bs(putcall, S, K, r, vol, dt_exp, dt_exdiv = today() + Day(1000), dt_divpmt = today() + Day(1000), divamt = .0000001, dt_trade = bd(today()))
	r = Rmm2Rcc(r)
	premium_settlement = 1
 	exercise_settlement = 2 #biz days
 	# adjust dates to good business day (following)
  	dt_trade = bd(dt_trade)
	dt_settle = workday(dt_trade, premium_settlement)
	dt_exp = bd(dt_exp)
	dt_exdiv = bd(dt_exdiv)
	dt_divpmt = bd(dt_divpmt)
	# now compute time differences (in days)
	t_exp = (dt_exp - dt_trade).value
	t_exdiv = (dt_exdiv - dt_trade).value
	t_divpmt = (dt_divpmt - dt_trade).value
	if t_exp < t_exdiv
		divamt = 0
	end
   	S0 = S - divamt * exp(-r * t_divpmt / 365)
	vol = vol * S / S0
	d1 = (log(S0/K) + (r + vol^2/2) * t_exp/365) / (vol * sqrt(t_exp/365))
	d2 = d1 - vol * sqrt(t_exp/365)
	dist = Normal()
	call = S0 * cdf(dist, d1) - K*exp(-r * t_exp/365) * cdf(dist, d2)
	put = call + exp(-r * t_exp/365) * K - S0
	call = call * exp(r * (dt_settle - dt_trade).value/365)
	put = put * exp(r * (dt_settle - dt_trade).value/365)
  	if lowercase(putcall)=="call"
		return call
	elseif lowercase(putcall)=="put"
		return put
	end
end

# ╔═╡ 1145e1de-2e0f-11eb-2e2a-2dccbfc0b2bd
function conversion_bs(S, K, r, vol, dt_exp, dt_exdiv = today() + Day(1000), dt_divpmt = today() + Day(1000), divamt = .0000001, dt_trade = bd(today()))
	
	callpx = option_bs("call",S,K,r,vol,dt_exp,dt_exdiv,dt_divpmt,divamt,dt_trade)
  	putpx = option_bs("put",S,K,r,vol,dt_exp,dt_exdiv,dt_divpmt,divamt,dt_trade)
  	conversion = putpx - callpx + S - K
  	return(conversion)           
end

# ╔═╡ f1a50ad8-2e10-11eb-3adc-2be63ff452c6
function implied_funding_bs(conversion_px, 
                   S,
                   K,
                   vol,
                   dt_exp,
                   dts_exdiv = today() + Day(1000),
                   dts_divpmt = today() + Day(1000),
                   divamts = 0.000001,
                   dt_trade = bd(today()))
	obj(x) = (conversion_bs(S,K,x,vol,dt_exp,dts_exdiv,dts_divpmt,divamts,dt_trade) - conversion_px)^2
	res = optimize(obj, -10, 10)
	r_mmm = Optim.minimizer(res)
	r_mmm
end

# ╔═╡ 046b49e0-3355-11eb-1d11-0ba9dde96a3b
let
	S=73.73
	K=75
	r= -3.005
	σ=.32
	dt_exp=Date(2020,12,24)
	N=1000
	dt_exdiv=Date(2021,12,19)
	dt_divpmt=Date(2021,12,19)
	divamt=0
	dt_trade=Date(2020,12,18)
	c_mkt = 4.45
	c = conversion(S,K,r,σ,dt_exp,N,dt_exdiv,dt_divpmt,divamt,dt_trade)
	c_e = conversion_bs(S,K,r,σ,dt_exp,dt_exdiv,dt_divpmt,divamt,dt_trade)
	ifa = implied_funding_american(c_mkt,S,K,σ,dt_exp,N,dt_exdiv,dt_divpmt,divamt,dt_trade)
	ife = implied_funding_bs(c_mkt,S,K,σ,dt_exp,dt_exdiv,dt_divpmt,divamt,dt_trade)
	@show c, c_e, ifa, ife
end

# ╔═╡ ffb0be7a-10a5-11eb-0b76-9588872dd222
function ema(x,n)
	a = x
	w = 2/(n+1)
	for i=2:length(x)
		a[i] = (1-w)*a[i-1] + w*x[i]
	end
	a
end

# ╔═╡ f344dda2-2cd1-11eb-1bee-bf2ddc84fe52
begin
	it = 1:10
	px = zeros(length(it));
	for n=1:length(it)
		px[n] = bintree("put", 94.9, 75, -3, 3, Date(2020,12,24), it[n], Date(2021,12,19), Date(2021,12,19), 2, Date(2020,12,21))[1]
	end
	g=Plots.plot(it,px,label="")
	pxbar = ema(px,20);
	plot!(g,it,pxbar,label=last(pxbar))
	xaxis!(g,"num steps")
	yaxis!(g,"price")
	g
end

# ╔═╡ 2b14bc28-4cbb-11eb-0bb6-f3df2f6f6ee0
option_bs("put",94.9,75,-3,3,Date(2020,12,24), Date(2021,12,19), Date(2021,12,19), 2, Date(2020,12,21))

# ╔═╡ ddfc9046-1466-11eb-022c-09091111cec8
let
	strikes = 50:5:150
	divs = 0:1:10
	S = 100
	r = 0.01
	σ = 0.4
	dt_exp = today() + Day(90)
	N = 1000
	dt_exdiv = today() + Day(2)
	dt_divpmt = today() + Day(2)
	dt_trade = today()
	step = 0.5
	
	deltas = [(conversion(S+step,strikes[i],r,σ,dt_exp,N,dt_exdiv,dt_divpmt, divs[j],dt_trade)[1,1] - conversion(S-step,strikes[i],r,σ,dt_exp,N,dt_exdiv,dt_divpmt, divs[j],dt_trade)[1,1]) / (2step) for j=1:length(divs), i=1:length(strikes)]
	contourf(strikes,divs, deltas)
	xaxis!("strike (% of spot)")
	yaxis!("dividend (% of spot)")
	title!("conversion Δ")
	#savefig("/home/alex/Downloads/mygraph.png")
end
		

# ╔═╡ e6278072-3415-11eb-14e1-6f4cdad78a4b
@bind go Button("Recompute")

# ╔═╡ 239373a2-3340-11eb-2d1c-dd42737ee049
let
	go
	S = 100
	divamt = S/10
	vol = .4
	r = .01
	strike = .8*S
	N = 500
	step = 0.5
	dt_exp = today() + Day(90)
	dt_exdiv = today() + Day(20)
	dt_divpmt = today() + Day(20)
	dt_trade = today()
	t_exp = (dt_exp - dt_trade).value
	t_exdiv = (dt_exdiv - dt_trade).value
	t_divpmt = (dt_divpmt - dt_trade).value
	tstep = 1#t_exp / N
	i_exdiv = t_exdiv / tstep
 	i_divpmt = t_divpmt / tstep
	R = exp(r * tstep / 365)
   	S0 = S - divamt / R^i_exdiv
	vol = vol * S / S0
	U = exp(vol * sqrt(tstep / 365))
	D = 1 / U
	p = (R - D) / (U - D)
	q = (U - R) / (U - D)
	sv = [S0]
	for i=1:(t_exp/tstep)
		if rand() <=p
			push!(sv,U*last(sv))
		else
			push!(sv,D*last(sv))
		end
	end
	sv[t_exdiv:end]=sv[t_exdiv:end] .- divamt
	conversion_deltapath = []
	call_deltapath = []
	put_deltapath = []
	for i=1:length(sv)
		x2 = conversion(sv[i]+step,strike,r,vol,dt_exp,N,dt_exdiv,dt_divpmt, divamt,dt_trade+Day(i-1))
		x1 = conversion(sv[i]-step,strike,r,vol,dt_exp,N,dt_exdiv,dt_divpmt, divamt,dt_trade+Day(i-1))
		push!(conversion_deltapath, (x2[1,1] - x1[1,1]) / (2step))
		push!(put_deltapath, (x2[2,1] - x1[2,1]) / (2step))
		push!(call_deltapath, (x2[3,1] - x1[3,1]) / (2step))
	end
	a = Plots.plot(sv,legend=false,yaxis="stock price")
	vline!(a,[t_exdiv],lw=2, lc = :black)
	b = Plots.plot(conversion_deltapath,legend=false,yaxis="Δ conversion")
	vline!(b,[t_exdiv],lw=2, lc = :black)
	c = Plots.plot(put_deltapath,legend=false)
	Plots.plot!(c,call_deltapath,legend=false)
	yaxis!(c, "Δ put,call")
	xaxis!(c,"days to expiry")
	vline!(c,[t_exdiv],lw=2, lc = :black)
	#ylabel!(c, ["Δ put" "Δ call"  "exdiv"])
	Plots.plot(a,b,c,layout=(3,1))
	#delplot
end

# ╔═╡ 01e292f8-4536-11eb-0795-79443fac8327
let
	if true
		S = 94.9
		divamt = 0
		vol = 2.5
		rmm = -3
		K = 75
		dt_exp = Date(2020,12,24)
		dt_exdiv = today() + Day(20)
		dt_divpmt = today() + Day(20)
		dt_trade = Date(2020,12,21)
	else
		S = 100
		divamt = 0.05 * S
		vol = .5
		rmm = .02
		K = 1.25 * S
		dt_exp = bd(today() + Day(90))
		dt_exdiv = bd(today() + Day(30))
		dt_divpmt = bd(today() + Day(40))
		dt_trade = today()
	end
	N = 1000
	step = 0.5
	r = Rmm2Rcc(rmm)
	percentages = collect(-.5:.05:.5)
	prices = (1 .+percentages) * S
	put_delta = zeros(length(prices))
	call_delta = zeros(length(prices))
	put_dte = zeros(length(prices))
	call_dte = zeros(length(prices))
	for i = 1:length(prices)
		S = prices[i]
		put_down = bintree("put", S-step, K, rmm, vol, dt_exp, N, dt_exdiv , dt_divpmt, divamt, workday(dt_trade,1))
		put_up = bintree("put", S+step, K, rmm, vol, dt_exp, N, dt_exdiv , dt_divpmt, divamt, workday(dt_trade,1))
		call_down = bintree("call", S-step, K, rmm, vol, dt_exp, N, dt_exdiv , dt_divpmt, divamt, workday(dt_trade,1))
		call_up = bintree("call", S+step, K, rmm, vol, dt_exp, N, dt_exdiv , dt_divpmt, divamt, workday(dt_trade,1))
		put_delta[i] = (put_up[1,1] - put_down[1,1])/(2step)
		call_delta[i] = (call_up[1,1] - call_down[1,1])/(2step)
		put_dte[i] = (put_up[3,1] + put_down[3,1])/2
		call_dte[i] = (call_up[3,1] + call_down[3,1])/2
	end
	
	
	spct=[]
	for i=1:length(percentages)
		push!(spct, string(round(Int,100*percentages[i]))*"%")
	end
	if false
		a = Plots.plot(percentages,put_delta,legend=false,yaxis="delta",title="PUT on t+1")
		b = Plots.plot(percentages,put_dte,legend=false,xaxis = "tomorrow's percentage change",yaxis="expected contract life (days)")
		c = Plots.plot(percentages,call_delta,legend=false,yaxis="delta",title="CALL on t+1")
		d = Plots.plot(percentages,call_dte,legend=false,xaxis = "tomorrow's percentage change",yaxis="expected contract life (days)")
		Plots.plot(a,c,b,d,layout=(2,2))
	else
		a = Plots.plot(percentages, 1 .+ put_delta - call_delta, legend = false,  yaxis = "delta", title = "CONVERSION on t+1")
		b = Plots.plot(percentages, min.(put_dte,call_dte),legend = false, xaxis = "tomorrow's percentage change", yaxis = "expected contract life (days)")
		Plots.plot(a,b,layout=(2,1))
	end
end

# ╔═╡ a716a79a-1468-11eb-1037-9b3919716663
function bintree_smooth(putcall, S, K, r, vol, dt_exp, dt_exdiv = today() + Day(1000), dt_divpmt = today() + Day(1000), divamt = .0000001, dt_trade = bd(today()))
	it = 300:1:400
	px = zeros(length(it));
	for i=1:length(it)
			px[i] = bintree(putcall, S, K, r, vol, dt_exp, it[i], dt_exdiv, dt_divpmt, divamt, dt_trade)[1]
	end
	last(ema(px,20))
end
			
			
	

# ╔═╡ e81a64e2-1469-11eb-0fc7-1bd0c6999999
bintree("put", 94.90, 75, -3, 3, Date(2020,12,24), 1000, Date(2021,12,19), Date(2021,12,19), 2, Date(2020,12,21))

# ╔═╡ 17e9ff34-146a-11eb-29f9-89746e7925d1
bintree_smooth("put", 94.9, 75, -3, 3, Date(2020,12,24), Date(2021,12,19), Date(2021,12,19), .000001, Date(2020,12,21))

# ╔═╡ Cell order:
# ╠═a2901126-3424-11eb-3e40-c76e36480a1a
# ╠═b580be6e-3424-11eb-0b25-718fabf3b6e4
# ╠═14726f12-4d5e-11eb-3f82-575c04cb690a
# ╟─cb315d1c-2da1-11eb-232b-3fcccc40e5e6
# ╟─983d5ef6-2d9c-11eb-260a-fbc85d734f14
# ╟─a4deccaa-449c-11eb-3342-eb6908001a34
# ╟─eb7f9626-449c-11eb-2b83-3f180644b7d1
# ╟─72a4d668-2d9e-11eb-26cf-d99bc088183f
# ╟─72177212-2da0-11eb-153e-4b8bf66c697c
# ╟─6ebf8d5e-2d97-11eb-204a-511e7aa6b14c
# ╠═9ef5f72c-2dd2-11eb-0d89-53e5346ab60b
# ╠═4ba87eea-4cad-11eb-1d51-ad3b81b92cfa
# ╠═dae9c4ba-4ca5-11eb-2c3d-75c98eb5d8b9
# ╠═e37960e8-4d38-11eb-048c-d5df65446524
# ╟─3394a1c6-2e0a-11eb-0847-6f070b42fc67
# ╟─d6747d5a-2eb2-11eb-1e13-670a197e8b1f
# ╟─1145e1de-2e0f-11eb-2e2a-2dccbfc0b2bd
# ╟─f1a50ad8-2e10-11eb-3adc-2be63ff452c6
# ╟─046b49e0-3355-11eb-1d11-0ba9dde96a3b
# ╟─ffb0be7a-10a5-11eb-0b76-9588872dd222
# ╟─f344dda2-2cd1-11eb-1bee-bf2ddc84fe52
# ╠═2b14bc28-4cbb-11eb-0bb6-f3df2f6f6ee0
# ╟─ddfc9046-1466-11eb-022c-09091111cec8
# ╟─239373a2-3340-11eb-2d1c-dd42737ee049
# ╟─e6278072-3415-11eb-14e1-6f4cdad78a4b
# ╟─01e292f8-4536-11eb-0795-79443fac8327
# ╟─a716a79a-1468-11eb-1037-9b3919716663
# ╠═e81a64e2-1469-11eb-0fc7-1bd0c6999999
# ╠═17e9ff34-146a-11eb-29f9-89746e7925d1
