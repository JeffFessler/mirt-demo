# 2D+T dynamic MRI simulation using GA radial sampling
# reconstructed with temporal "TV" regularizer (corner-rounded)
# 2019-06-13, Jeff Fessler

using MIRT
using Plots: gui, plot, scatter
using LinearAlgebra: norm, Diagonal
using Random: seed!

# generate dynamic image sequence
if !@isdefined(xtrue)
	N = (60,64)
	fov = 220
	nt = 8 # frames
	ig = image_geom(nx=N[1], ny=N[2], fov=fov)

	_,ellpar = ellipse_im(ig, :southpark, return_params=true)
	ellpars = Array{Float32}(undef, size(ellpar)..., nt)
	xtrue = Array{ComplexF32}(undef, N..., nt)
	for it=1:nt
		tmp = ellpar
		tmp[2,4] = 15 + 5 * sin(2*pi*it/nt) # mouth open/close
		ellpars[:,:,it] = tmp
		xtrue[:,:,it] = ellipse_im(ig, ellpars[:,:,it], oversample=4)
		jim(ig.x(), ig.y(), xtrue[:,:,it], title="frame $it")
		gui()
	end
end

# plot one time course to see temporal change
# plot(1:nt, abs.(xtrue[30,14,:]))

# k-space sampling and data
if !@isdefined(kspace)
	@show accelerate = 3
	@show nspf = round(Int, maximum(N)/accelerate) # spokes per frame
	Nro = maximum(N)
	Nspoke = nspf * nt
	kspace = ir_mri_kspace_ga_radial(Nro = Nro, Nspoke = Nspoke)
	kspace ./= ig.fov
	kspace = reshape(kspace, Nro, nspf, nt, 2)
end

if false # plot sampling
	ps = Array{Any}(undef, nt)
	for it=1:nt
		ps[it] = scatter(kspace[:,:,it,1], kspace[:,:,it,2], aspect_ratio=1, label="")
		plot(ps[it])
		gui()
	end
#	plot(ps...)
end


# make system matrix for dynamic non-Cartesian parallel MRI
if !@isdefined(A)
	# a NUFFT object for each frame
	ns = Array{Any}(undef, nt)
	for it=1:nt
		om = [kspace[:,:,it,1][:] kspace[:,:,it,2][:]] * fov * 2 * pi
		ns[it] = nufft_init(om, N, n_shift = collect(N)/2)
	end

	# block diagonal system matrix, with one NUFFT per frame
	# todo: more realistic coil sensitivity
	ncoil = 2
	S = [Diagonal(ic*ones(Float32, N)[:]) for ic=1:ncoil] # silly sensitivity maps
	Slm = block_lm(S, how=:col) # [S1; ... ; Sncoil]
	mykron = A1 -> block_lm([A1], how=:kron, Mkron=ncoil) # one NUFFT per coil

	# output is essentially [nt Ncoil nspf Nro] (which is possibly unusual)
	# input is [N... nt]
	A = block_lm([mykron(s.A)*Slm for s in ns], how=:diag)
end


# simulate k-space data via an inverse crime
if !@isdefined(y)
end
	ytrue = A * xtrue[:]

	snr2sigma = (db, yb) -> 
		exp(-db/20) * norm(yb) / sqrt(length(yb)) # / sqrt(2) # for complex noise

	sig = snr2sigma(80, ytrue)
	seed!(0)
	y = ytrue + sig * randn(ComplexF32, size(ytrue))
	y = ytrue + sig * randn(Float32, size(ytrue))
	@show 20*log10(norm(ytrue) / norm(y - ytrue))


if false
	# todo: need density compensation, perhaps via
	# https://github.com/JuliaGeometry/VoronoiDelaunay.jl
	tmp = reshape(A' * y, N..., nt) # zero-filled recon
	jim(tmp)
	gui()
end

if false

"""
`stackpick(x::AbstractArray{<:Any}, i::Int)`
return `x[:,...,:,i]`
"""
function stackpick(x::AbstractArray{<:Any}, i::Int)
	xdim = size(x)
	dim1 = xdim[1:end-1]
	return reshape((@view reshape(x, prod(dim1), xdim[end])[:,i]), dim1) 
end

	forw = x -> begin
			out = Array{Any}(undef, nt)
			for it=1:nt
				out[it] = ns[it].nufft(stackpick(x,it))
			end
			return vcat(out...)
		end
	tmp = forw(xtrue)
	@show size(tmp)
end
