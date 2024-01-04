import matplotlib.pyplot as plt
import seaborn as sns

def make_slice_plots(eta_true, writer, i, title): 
	res = eta_true.shape[0]
	eta_slice = eta_true.reshape(res, res, 8, -1).mean(3)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res-1])
		ax.set_ylim([0, res-1])
		ax.set_title(title+' Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
		plt.colorbar(heatmap_im, ax=ax)
	if writer is not None:
		writer.add_figure(title+' Slice Plot', fig, i)
	else: 
		plt.show()
	plt.close()

def make_slice_plots_lum(eta_true, writer, i, title, vmin=True): 
	eta_slice = eta_true.reshape(64, 64, -1, 8).mean(3)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, 64])
		ax.set_ylim([0, 64])
		ax.set_title(title+' Field Slice %d'%j)
		if vmin: 
			heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
		else: 
			heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=eta_true.min(), vmax=eta_true.max())
		plt.colorbar(heatmap_im, ax=ax)
	# plt.savefig(save_dir + 'true_eta_slice.png')
	if writer is not None:
		writer.add_figure(title+' Slice Plot', fig, i)
	else: 
		plt.show()
	plt.close()

def heatmap_2d(image, title): 
	fig, ax = plt.subplots()
	res = image.shape[0]
	plt.xlim([0, res-1])
	plt.ylim([0, res-1])
	heatmap_im = ax.imshow(image)
	plt.colorbar(heatmap_im, ax=ax)
	plt.title(title)

	return fig

def plots_regular(res, ray_lum_plot, ray_lum_target_plot, eta, eta_true, writer, i): 
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21., 5.))
	# plot ray image
	ax1.set_xlim([0, res-1])
	ax1.set_ylim([0, res-1])
	ax1.set_title('Model Luminance Image')
	heatmap_im = ax1.imshow(ray_lum_plot.reshape(64, 64)[::-1, :].T, vmin=0., vmax=ray_lum_target_plot.max())
	plt.colorbar(heatmap_im, ax=ax1)

	# plot ray image target
	ax2.set_xlim([0, res-1])
	ax2.set_ylim([0, res-1])
	ax2.set_title('Target Luminance Image')
	heatmap_im = ax2.imshow(ray_lum_target_plot.reshape(64, 64)[::-1, :].T, vmin=0., vmax=ray_lum_target_plot.max())
	plt.colorbar(heatmap_im, ax=ax2)

	# plot image difference
	ax3.set_xlim([0, res-1])
	ax3.set_ylim([0, res-1])
	ax3.set_title('Luminance Image Diff (Model - Target)')
	heatmap_im = ax3.imshow((ray_lum_plot - ray_lum_target_plot).reshape(64, 64)[::-1, :].T)
	plt.colorbar(heatmap_im, ax=ax3)
	writer.add_figure('Image Plot', fig, i)
	plt.close()

	# plot eta field slices
	# eta field should be 16 x 16 x 16
	res_eta = eta_true.shape[0]
	eta_slice = eta.reshape(res_eta, res_eta, 8, -1).mean(3)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot', fig, i)
	plt.close()

	# plot eta field slices (colorbar not fixed)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=eta_slice.min(), vmax=eta_slice.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot Orig Scale', fig, i)
	plt.close()

def plots_eta(res, ray_lum_plot, ray_lum_target_plot, eta, eta_true, writer, i): 

	# plot eta field slices
	res_eta = eta_true.shape[0]
	eta_slice = eta.reshape(res_eta, res_eta, 8, -1).mean(3)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot', fig, i)
	plt.close()

	# plot eta field slices (colorbar not fixed)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=eta_slice.min(), vmax=eta_slice.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot Orig Scale', fig, i)
	plt.close()

def plots_img(eta, render_img, target_img, eta_true, writer, i):
	res = 64 
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21., 5.))
	# plot ray image
	ax1.set_title('Model Image')
	sns.heatmap(render_img.reshape(64, 64).reshape(res, res)[::-1, ::-1].T, ax=ax1)

	# plot ray image target
	ax2.set_title('Target Image')
	sns.heatmap(target_img.reshape(64, 64).reshape(res, res)[::-1, ::-1].T, ax=ax2)

	# plot image difference
	ax3.set_title('Image Diff (Model - Target)')
	sns.heatmap((render_img - target_img).reshape(64, 64).reshape(res, res)[::-1, ::-1].T, ax=ax3)
	writer.add_figure('Image Plot', fig, i)
	plt.close()

	# plot eta field slices
	# eta field should be 16 x 16 x 16
	res_eta = eta_true.shape[0]
	eta_slice = eta.reshape(res_eta, res_eta, 8, -1).mean(3)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot', fig, i)
	plt.close()

	# plot eta field slices (colorbar not fixed)
	fig, axes = plt.subplots(2, 4, figsize=(28., 10.))
	for j in range(8): 
		k = j % 4
		l = j // 4
		ax = axes[l, k]
		ax.set_xlim([0, res_eta-1])
		ax.set_ylim([0, res_eta-1])
		ax.set_title('Eta Field Slice %d'%j)
		heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=eta_slice.min(), vmax=eta_slice.max())
		plt.colorbar(heatmap_im, ax=ax)
	writer.add_figure('Eta Slice Plot Orig Scale', fig, i)
	plt.close()