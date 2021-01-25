### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 2ad8b1a4-5cbc-11eb-3e93-91e93e9c5537
begin
	using Pkg; Pkg.activate("../random")
	using Images, ImageIO, ImageFiltering, PlutoUI, ImageView, Plots, Random, StatsBase
end

# ╔═╡ 43306da6-5f3d-11eb-09a5-79c47d8e603a
begin
	image = load("../2.png")
	image = imresize(image, ratio = 1/4)
	gen_size = 50
	num_iter = 100
end

# ╔═╡ 4d38fd7a-5f3d-11eb-3f73-2b2698b7ed1d
function vectorMSEloss(a,b)
	return sum((a-b).^2)
end

# ╔═╡ 73e1def6-5f3d-11eb-215f-d9c6bb5c620d
function choose_eta(size)
	basis_ind = StatsBase.sample(collect(1:size[0]), size[1], replace = false)
	basis = zeros(size)
	basis[basis_ind, 1:size[1]] = 1
	return basis
end

# ╔═╡ 503d96d4-5f3d-11eb-071a-53d8f8996788
function dogenerations(image::Array{ColorTypes.RGB{FixedPointNumbers.Normed{UInt8,8}},2},
		gen_size::Int64, 
		num_iter::Int64)
	lr = 0.5
	
	guidict = ImageView.imshow(image)
	sleep(0.1)
	
	canvas = guidict["gui"]["canvas"]
	w,h = size(image)
	image = reshape(vec(channelview(image)), (*(w,h,3),1))
	images = randn((*(3,h,w), gen_size))
	iter_loss = []
	
	for i in 1:num_iter
		losses = vectorMSEloss(image, images[:,1])
		
		for j in 2:size(images)[2]
			losses = [losses vectorMSEloss(image, images[:,j])]
		end
		losses = sort(collect(zip(losses, 1:gen_size)))
		images = images[:, [losses[j][2] for j in 1:(gen_size ÷ 2)]]

		pix_error = images .- image
		noise = pix_error.*randn((size(image)[1], gen_size ÷ 2))
		gen_images = clamp.(images.+noise, 0,1)
		images = [images gen_images]
		ImageView.imshow(canvas, colorview(RGB, reshape(images[:,1], (3, w,h))))
		println(i,losses[1])
		if i == 1
			iter_loss = [losses[1][1][1]]
		else
			iter_loss = append!(iter_loss,losses[1][1][1])
		end
	end
	Plots.plot(1:num_iter, iter_loss, size = (1500, 1500))
end

# ╔═╡ bc661222-5cc3-11eb-0018-c1554178593f
StatsBase.sample(collect(1:100), 10, replace = false)

# ╔═╡ Cell order:
# ╠═2ad8b1a4-5cbc-11eb-3e93-91e93e9c5537
# ╠═43306da6-5f3d-11eb-09a5-79c47d8e603a
# ╠═4d38fd7a-5f3d-11eb-3f73-2b2698b7ed1d
# ╠═73e1def6-5f3d-11eb-215f-d9c6bb5c620d
# ╠═503d96d4-5f3d-11eb-071a-53d8f8996788
# ╠═bc661222-5cc3-11eb-0018-c1554178593f
