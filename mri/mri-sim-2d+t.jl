# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Julia 1.4.2
#     language: julia
#     name: julia-1.4
# ---

# ### 2D+T dynamic MRI simulation
# using GA radial sampling
# reconstructed with temporal "TV" regularizer (corner-rounded)  
# 2019-06-13, Jeff Fessler  
# 2019-06-23 update to use more realistic simulated sensitivity maps  
# 2020-06-20 update

using MIRT: jim, prompt
using MIRT: image_geom, ellipse_im_params, ellipse_im
using MIRT: nufft_init, diffl_map, ncg
using MIRT: ir_mri_sensemap_sim, ir_mri_kspace_ga_radial
using Plots: gui, plot, scatter, default; default(markerstrokecolor=:auto)
using LinearAlgebra: norm, dot, Diagonal
using LinearMapsAA
using Random: seed!
jim(:abswarn, false); # suppress warnings about display of |complex| images


# generate dynamic image sequence
if !@isdefined(xtrue)
	N = (60,64)
	fov = 220
	nt = 8 # frames
	ig = image_geom(nx=N[1], ny=N[2], fov=fov)

	ellpar = ellipse_im_params(ig, :southpark)
	ellpars = Array{Float32}(undef, size(ellpar)..., nt)
	xtrue = Array{ComplexF32}(undef, N..., nt)
	for it=1:nt
		tmp = ellpar
		tmp[2,4] = 15 + 5 * sin(2*pi*it/nt) # mouth open/close
		ellpars[:,:,it] = tmp
		xtrue[:,:,it] = ellipse_im(ig, ellpars[:,:,it], oversample=4)
		jim(ig.x, ig.y, xtrue[:,:,it], title="frame $it")
		gui()
	end
end
jim(xtrue, yflip=ig.dy < 0)
#prompt()


# plot one time course to see temporal change
ix,iy = 30,14
plot(1:nt, abs.(xtrue[ix,iy,:]), label="ix=$ix, iy=$iy",
	marker=:o, xlabel="frame")
#prompt()


# +
# k-space sampling and data
if !@isdefined(kspace)
	@show accelerate = 3
	@show nspf = round(Int, maximum(N)/accelerate) # spokes per frame
	Nro = maximum(N)
	Nspoke = nspf * nt
	kspace = ir_mri_kspace_ga_radial(Nro = Nro, Nspoke = Nspoke)
	kspace[:,:,1] ./= ig.fovs[1]
	kspace[:,:,2] ./= ig.fovs[2]
	kspace = reshape(kspace, Nro, nspf, nt, 2)

	if true # plot sampling (in units of cycles/pixel)
		ps = Array{Any}(undef, nt)
		for it=1:nt
			ps[it] = scatter(kspace[:,:,it,1] * ig.fovs[1], kspace[:,:,it,2] * ig.fovs[2],
				xtick=(-1:1)*0.5, ytick=(-1:1)*0.5, xlim=[-1,1]*0.52, ylim=[-1,1]*0.52,
				aspect_ratio=1, label="", markersize=1)
			plot(ps[it])
			gui()
		end
		plot(ps..., layout=(2,4))
	#	prompt()
	end
end
# -


#+
# make sensitivity maps, normalized so SSoS = 1
if !@isdefined(smap)
	ncoil = 2
	smap = ir_mri_sensemap_sim(dims=N, ncoil=ncoil, orbit_start=[90])
	p1 = jim(smap, "ncoil=$ncoil sensitivity maps raw")

	ssos = sqrt.(sum(abs.(smap).^2, dims=ndims(smap))) # SSoS
	ssos = selectdim(ssos, ndims(smap), 1)
	p2 = jim(ssos, "SSoS")

	for ic=1:ncoil
		selectdim(smap, ndims(smap), ic) ./= ssos
	end
	p3 = jim(smap, "ncoil=$ncoil sensitivity maps")

	ssos = sqrt.(sum(abs.(smap).^2, dims=ndims(smap))) # SSoS
	@assert all(isapprox.(ssos,1))
	plot(p1, p2, p3)
#	prompt()
end
#-


#+
# make system matrix for dynamic non-Cartesian parallel MRI
if !@isdefined(A)
	# a NUFFT object for each frame
	ns = Array{Any}(undef, nt)
	for it=1:nt
		om = [kspace[:,:,it,1][:] kspace[:,:,it,2][:]] * fov * 2 * pi
		ns[it] = nufft_init(om, N, n_shift = collect(N)/2)
	end

	# block diagonal system matrix, with one NUFFT per frame
	S = [Diagonal(selectdim(smap, ndims(smap), ic)[:]) for ic=1:ncoil]
	SO = s -> LinearMapAA(s ; idim=N, odim=N) # LinearMapAO
	AS1 = A1 -> vcat([A1 * SO(s) for s in S]...) # [A1*S1; ... ; A1*Sncoil]

	# output is essentially [nt Ncoil nspf Nro] (which is possibly unusual)
	# input is [N... nt]
	A = block_diag([AS1(s.A) for s in ns]...)
end
#-


#+
# simulate k-space data via an inverse crime
if !@isdefined(y)
	ytrue = A * xtrue

	snr2sigma = (db, yb) -> # compute noise sigma from SNR (no sqrt(2) needed)
		10^(-db/20) * norm(yb) / sqrt(length(yb))

	sig = Float32(snr2sigma(50, ytrue))
	seed!(0)
	y = ytrue + sig * randn(ComplexF32, size(ytrue))
	@show 20*log10(norm(ytrue) / norm(y - ytrue)) # verify SNR
end
#-


# initial image via zero-fill and scaling
if !@isdefined(x0)
	# todo: should use density compensation, perhaps via
	# https://github.com/JuliaGeometry/VoronoiDelaunay.jl
	x0 = A' * y # zero-filled recon (for each frame)
	tmp = A * x0 # Nkspace × Ncoil × Nframe
	x0 = (dot(tmp,y) / norm(tmp)^2) * x0 # scale sensibly
	jim(x0, "initial image", yflip=ig.dy < 0)
end
#prompt()


# +
# temporal finite differences
if !@isdefined(Dt)
	Dt = diffl_map((N..., nt), length(N)+1 ; T=eltype(A))
	tmp = Dt' * (Dt * xtrue)
	jim(tmp, "time diff", yflip=ig.dy < 0)
end
#prompt()
# -


# +
# run nonlinear CG
if !@isdefined(xh)
	niter = 90
	delta = Float32(0.1) # small relative to temporal differences
	reg = Float32(2^20) # trial and error here
	ffair = (t,d) -> d^2 * (abs(t)/d - log(1 + abs(t)/d))
	pot = z -> ffair(z, delta)
	dpot = z -> z / (Float32(1) + abs(z/delta))
	cost = x -> 0.5 * norm(A*x - y)^2 + reg * sum(pot.(Dt * x))
	fun = (x,iter) -> cost(x)
	gradf = [v -> v - y, u -> reg * dpot.(u)]
	curvf = [v -> Float32(1), u -> reg]
	B = [A, Dt]
	(xh, out) = ncg(B, gradf, curvf, x0 ; niter=niter, fun=fun)
	costs = [out[i+1][1] for i=0:niter]
end

# show results
	plot(layout=(2,2),
		jim(xtrue, "xtrue", yflip=ig.dy < 0),
		jim(xh, "recon", yflip=ig.dy < 0),
		jim(xh-xtrue, "error", yflip=ig.dy < 0),
		scatter(0:niter, log.(costs), label="cost", xlabel="iteration"),
	)
