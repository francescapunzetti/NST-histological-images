import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def compare_images(imageA, imageB, title):
	# compute the structural similarity index for the images
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("SSIM: %.2f" % (s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

# load the images: the original and the AI generated
original = cv2.imread("images/internal_colored.jpg")
new = cv2.imread("images/im 2/int_800.jpg")
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

fig = plt.figure("Images")
images = ("Original", original), ("AI generated", new)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()
# compare the images
compare_images(original, new, "Original vs. AI generated")