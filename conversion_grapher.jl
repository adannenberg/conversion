### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a2901126-3424-11eb-3e40-c76e36480a1a
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ b580be6e-3424-11eb-0b25-718fabf3b6e4
begin
	Pkg.add(["Plots","Dates", "Plotly","PlutoUI"])
	using Plots, Dates, PlutoUI
	plotly()
end

# ╔═╡ 14726f12-4d5e-11eb-3f82-575c04cb690a
html"""<style>
main {
    max-width: 1900px;
}
"""

# ╔═╡ 9f8efa74-4d73-11eb-00c6-35d6ebb53290
md"
Change the following parameters to define the conversion whose pnl and early exercise we want to investigate.

When you modify an input you will see the cell's left-hand border turn red.  This means that you still have to 
'submit the changes' by holding down **Shift** and pressing **Enter**.  

In other words, Shift-Enter is like F9 in Excel
when autocalc is not on...
"

# ╔═╡ 79502d9e-4d71-11eb-3010-abbca2d3e4f9
begin
	S= 100
	K= 120
	r_stock = -.01
	r_option = .01
	vol = .3
	dt_exp = today() + Day(90)
	dt_exdiv = today() + Day(30)
	dt_divpmt = today() + Day(40)
	dt_trade = today()
	#= dates can be written generally or specifically, i.e. like
	date = today() + Day(9) to represent a day that's 9 days from today
	or
	date = Date(2021,2,17) for Feb 17, 2021
	=#
	divamt = .37
end

# ╔═╡ 4f41e9fa-4d73-11eb-070d-454fcdc14175
md"""
N.B. the conversion fuction returns a triplet = (conversion price, put price, call price):
"""

# ╔═╡ f885f88a-4d73-11eb-0444-6328098f4b92


# ╔═╡ f9d5fbd6-4d73-11eb-3b8b-0d54ba7e92db


# ╔═╡ fac19d8e-4d73-11eb-0f19-9b0a4d3c2420


# ╔═╡ fb9acc76-4d73-11eb-3d1e-0df22599c83f


# ╔═╡ dc728f58-4d66-11eb-1608-c5d4ccb69678
us_trading_holidays_2021=[Date(2020,11,26),Date(2020,12,25),Date(2021,1,1),Date(2021,1,18),Date(2021,2,15),Date(2021,4,2),Date(2021,5,31),Date(2021,7,5),Date(2021,9,6),Date(2021,11,25),Date(2021,12,24)]

# ╔═╡ 416c381e-4d67-11eb-3b3a-d547db43a309
us_trading_holidays_2022=[Date(2022,1,17),Date(2022,2,21),Date(2022,4,15),Date(2022,5,30),Date(2022,7,4),Date(2022,9,5),Date(2022,11,24),Date(2022,12,26)]

# ╔═╡ 99bdb5ec-4d67-11eb-19e3-e54fe2496452
us_trading_holidays=vcat(us_trading_holidays_2021,us_trading_holidays_2022)

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

# ╔═╡ f0f688c6-4d6b-11eb-307a-3b3e3cbca205
conversion(S,K,r_option,vol,dt_exp,500,dt_exdiv,dt_divpmt,divamt,dt_trade,false)

# ╔═╡ dae9c4ba-4ca5-11eb-2c3d-75c98eb5d8b9
function mygrapher(S,
                   K,
                   r_stock,
                   r_option,
                   vol,
                   dt_exp,
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
	
	p1 = scatter(boundary_info[:,1],boundary_info[:,4],xlims = (0,t_exp),label="put exercise bndry",title = "early exercise boundaries")
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

	# replace 0 values for cols 3,5 by missings so they dont get graphed with pnl
	for c=[3,5]
		indx = findall(boundary_info[:,c] .< .001)
		if length(indx) > 0
			boundary_info[indx,c] .= NaN
		end
	end
	pnl_upper_bound = pnl + boundary_info[:,3]
	pnl_lower_bound = pnl - boundary_info[:,5]
	#if all the values in col 3, i.e. all the put prices at the call exercise bndry, are NaNs then pnl will equal the upper bound
	# conversely if all the values in col 5 are NaNs then pnl will equal it's lower bound
	p2lab=""
	if length(findall(isfinite.(boundary_info[:,3]))) == 0
		p2lab = "pnl upper bound"
	elseif length(findall(isfinite.(boundary_info[:,5]))) == 0
		p2lab = "pnl lower bound"
	end
	p2 = plot(boundary_info[:,1],pnl,label=p2lab,title="pnl per share as of exercise dt")
    plot!(p2,boundary_info[:,1],pnl_upper_bound,label="pnl upper bound",title="pnl per share as of exercise dt")
	plot!(p2,boundary_info[:,1],pnl_lower_bound,label="pnl lower bound")
	p3 = scatter(t,boundary_info[:,3],label = "put price at call bndry",title = "put price at call bndry")
	p4 = scatter(t,boundary_info[:,5],label = "call price at put bndry",title = "call price at put bndry")
	plot(p2,p1,p3,p4,layout=(4,1),legend=false,size=(600,1200))
end


# ╔═╡ 4ba87eea-4cad-11eb-1d51-ad3b81b92cfa
mygrapher(S,K,r_stock,r_option,vol,dt_exp,dt_exdiv,dt_divpmt,divamt,dt_trade)

# ╔═╡ Cell order:
# ╟─a2901126-3424-11eb-3e40-c76e36480a1a
# ╟─b580be6e-3424-11eb-0b25-718fabf3b6e4
# ╟─14726f12-4d5e-11eb-3f82-575c04cb690a
# ╟─9f8efa74-4d73-11eb-00c6-35d6ebb53290
# ╠═79502d9e-4d71-11eb-3010-abbca2d3e4f9
# ╟─4ba87eea-4cad-11eb-1d51-ad3b81b92cfa
# ╟─4f41e9fa-4d73-11eb-070d-454fcdc14175
# ╟─f0f688c6-4d6b-11eb-307a-3b3e3cbca205
# ╟─f885f88a-4d73-11eb-0444-6328098f4b92
# ╟─f9d5fbd6-4d73-11eb-3b8b-0d54ba7e92db
# ╟─fac19d8e-4d73-11eb-0f19-9b0a4d3c2420
# ╟─fb9acc76-4d73-11eb-3d1e-0df22599c83f
# ╟─dc728f58-4d66-11eb-1608-c5d4ccb69678
# ╟─416c381e-4d67-11eb-3b3a-d547db43a309
# ╟─99bdb5ec-4d67-11eb-19e3-e54fe2496452
# ╟─983d5ef6-2d9c-11eb-260a-fbc85d734f14
# ╟─a4deccaa-449c-11eb-3342-eb6908001a34
# ╟─eb7f9626-449c-11eb-2b83-3f180644b7d1
# ╟─72a4d668-2d9e-11eb-26cf-d99bc088183f
# ╟─72177212-2da0-11eb-153e-4b8bf66c697c
# ╟─9ef5f72c-2dd2-11eb-0d89-53e5346ab60b
# ╟─dae9c4ba-4ca5-11eb-2c3d-75c98eb5d8b9
