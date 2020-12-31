from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from skimage import io

IMAGE_FILEPATH = "C:/Users/trgre/Desktop/Programming/Python/PaintByNumber/"
FLOOD_FILL_TOLERANCE = 100 #color difference squared that creates the initial flood fill image
SMALL_CELL_TOLERANCE = 40 #number of pixels to determine initially if set is too small
PASS_TWO_COLOR_TOLERANCE = 300 #color difference squared that creates the secondary merge of similarly colored cells

class ImageData:
	def __init__(self, image):
		original = Image.open(image)
		self.width, self.height = original.size #size of the image
		self.pixel_map = {} #maps from pixel location to LAB color code tuple
		self.pixel_cells_map = {} #maps from pixel location to the cell number
		self.unassigned_pixels = set() #holds the pixels that have not been assigned to a set
		self.filled_sets = {} #holds the sets that are grouped together by color
		i=0
		while(i<self.width):
			j=0
			while(j<self.height):
				self.pixel_map[(i, j)] = rgb2lab(original.getpixel((i, j)))
				self.unassigned_pixels.add((i, j))
				j += 1
			i += 1
	
	def floodfill(self, start_pixel, cellnumber):
		color = self.pixel_map[start_pixel]
		group = set()
		neighbors = set()
		current = start_pixel
		while(True):
			if(colordif(color, self.pixel_map[current]) < FLOOD_FILL_TOLERANCE):
				group.add(current)
				if(current[0]+1 <= self.width and (current[0]+1, current[1]) in self.unassigned_pixels):
					neighbors.add((current[0]+1, current[1]))
					self.unassigned_pixels.remove((current[0]+1, current[1]))
				if(current[0]-1 >= 0 and (current[0]-1, current[1]) in self.unassigned_pixels):
					neighbors.add((current[0]-1, current[1]))
					self.unassigned_pixels.remove((current[0]-1, current[1]))
				if(current[1]+1 <= self.height and (current[0], current[1]+1) in self.unassigned_pixels):
					neighbors.add((current[0], current[1]+1))
					self.unassigned_pixels.remove((current[0], current[1]+1))
				if(current[1]-1 >= 0 and (current[0], current[1]-1) in self.unassigned_pixels):
					neighbors.add((current[0], current[1]-1))
					self.unassigned_pixels.remove((current[0], current[1]-1))
			else:
				self.unassigned_pixels.add(current)
			if(len(neighbors) == 0):
				break
			current = neighbors.pop()
		self.filled_sets[cellnumber] = group
		for pixel in group:
			self.pixel_cells_map[pixel] = cellnumber
		return
		
	def smallcells(self, tolerance):
		while(True):
			small_cells = {}
			for cellnumber, cell in self.filled_sets.items():
				if(len(cell) <= tolerance):
					cell_neighbors = self.maxneighborcell(cellnumber)
					closest_cell = max(cell_neighbors, key=cell_neighbors.count)
					small_cells[cellnumber] = closest_cell
			print(str(len(small_cells)) + " small cells identified under a tolerance of " + str(tolerance) + " pixels")
			if(len(small_cells) == 0):
				break
			print("Removing small cells...")
			for small, large in small_cells.items():
				if(large not in self.filled_sets):
					continue
				for pixel, setnum in self.pixel_cells_map.items():
					if(setnum == small):
						self.filled_sets[large].add(pixel)
						self.pixel_cells_map[pixel] = large
				self.filled_sets.pop(small)
				
	def mergesimilar(self, color_tolerance):
		cell_mean = self.meancolors()
		largest_smallest = []
		for k in sorted(self.filled_sets, key=lambda k:len(self.filled_sets[k]), reverse=True):
			largest_smallest.append(k)
		for cellnum in largest_smallest:
			if(cellnum not in self.filled_sets):
				continue
			neighbor_cells = self.maxneighborcell(cellnum)
			neighbor_cells = list(dict.fromkeys(neighbor_cells))
			for neighbor in neighbor_cells:
				if(colordif(cell_mean[cellnum], cell_mean[neighbor]) < color_tolerance):
					for pixel, setnum in self.pixel_cells_map.items():
						if(setnum == neighbor):
							self.filled_sets[cellnum].add(pixel)
							self.pixel_cells_map[pixel] = cellnum
					self.filled_sets.pop(neighbor)
	
	def bordercolored(self, file):
		lines = Image.new("RGB", (self.width, self.height))
		for cellnum, set in self.filled_sets.items():
			for pixel in set:
				for item in [(pixel[0]+1, pixel[1]), (pixel[0]-1,pixel[1]), (pixel[0], pixel[1]+1), (pixel[0], pixel[1]-1)]:
					if(item[0] >= 0 and item[0] < self.width and item[1] >= 0 and item[1] < self.height):
						if(item not in set):
							lines.putpixel(item, (0, 0, 0))
						else:
							lines.putpixel(item, (255, 255, 255))
		return lines
		
	def addnumbers(self, file, line_image, painted_image, colors):
		code = [[[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,1,1,1,0],[1,1,0,1,1,1,1,0],[1,1,1,1,0,1,1,0],[1,1,1,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0]],
					[[0,0,0,1,1,0,0,0],[0,0,1,1,1,0,0,0],[0,1,1,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,1,1,1,1,1,1,0]],
					[[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,1,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,1,1,0],[1,1,1,1,1,1,1,0]],
					[[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0]],
					[[0,0,0,0,1,1,0,0],[0,0,0,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,0,1,1,0,0],[1,1,0,0,1,1,0,0],[1,1,1,1,1,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,1,1,0]],
					[[1,1,1,1,1,1,1,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0]],
					[[0,0,1,1,1,0,0,0],[0,1,1,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0]],
					[[1,1,1,1,1,1,1,0],[1,1,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0]],
					[[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,0,0]],
					[[0,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,1,1,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,1,1,1,1,0,0,0]],
					[[0,0,0,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,1,1,0,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,1,1,1,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0]],
					[[1,1,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[1,1,1,1,1,1,0,0]],
					[[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,0,0,0,0,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
					[[1,1,1,1,1,0,0,0],[0,1,1,0,1,1,0,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,1,1,0,0],[1,1,1,1,1,0,0,0]],
					[[1,1,1,1,1,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,0,1,0],[0,1,1,0,1,0,0,0],[0,1,1,1,1,0,0,0],[0,1,1,0,1,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,1,0],[0,1,1,0,0,1,1,0],[1,1,1,1,1,1,1,0]],
					[[1,1,1,1,1,1,1,0],[0,1,1,0,0,1,1,0],[0,1,1,0,0,0,1,0],[0,1,1,0,1,0,0,0],[0,1,1,1,1,0,0,0],[0,1,1,0,1,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,1,0,0,0,0,0],[1,1,1,1,0,0,0,0]]]
		numbers = Image.new("RGB", (2 * self.width, 2 * self.height))
		for i in range(0, 2*self.width):
			for j in range(0, 2*self.height):
				numbers.putpixel((i, j), line_image.getpixel((i/2, j/2)))
		for cellnum, set in self.filled_sets.items():
			x_pixels = np.array([pixel[0] for pixel in set])
			y_pixels = np.array([pixel[1] for pixel in set])
			x_mode = stats.mode(x_pixels).mode[0]
			y_mode = stats.mode(y_pixels).mode[0]
			x = x_mode
			y = y_mode
			while (x, y) not in set:
				if(x != 0):
					x -= 1
				elif(y != 0):
					x = x_mode
					y -= 1
				else:
					x = x_mode
					y = y_mode
					break
			x_mode = x
			y_mode = y
			if(x_mode <= 4):
				x_mode = 4
			elif(x_mode >= (2*self.width)-4):
				x_mode = (2*self.width)-4
			if(y_mode <= 5):
				y_mode = 5
			elif(y_mode >= (2*self.height)-7):
				y_mode = (2*self.height)-7
			color_code = 0
			pixel = (0, 0)
			for p in set:
				pixel = p
				break
			for option in colors:
				if(painted_image.getpixel(pixel) == option).all():
					break
				else:
					color_code += 1
			for i in range(0, 8):
				for j in range(0, 10):
					if(code[color_code][j][i] == 1 and ((2*x_mode) - 4 + i) >= 0 and ((2*x_mode) - 4 + i) < (2*self.width) and ((2*y_mode) - 5 + j) >= 0 and ((2*y_mode) - 5 + j) < (2*self.height)):
						numbers.putpixel(((2*x_mode) - 4 + i, (2*y_mode) - 5 + j), (255, 75, 75))
		numbers.save("C:/Users/trgre/Desktop/Programming/Python/PaintByNumber/coded_" + file)
		plt.imshow(numbers)
		plt.show()
			
	
	def outputimage(self, file):
		out = Image.new("RGB", (self.width, self.height))
		colors = self.meancolors()
		for cellnum, set in self.filled_sets.items():
			for pixel in set:
				out.putpixel(pixel, lab2rgb(colors[cellnum]))
		out.save("C:/Users/trgre/Desktop/Programming/Python/PaintByNumber/grouped_" + file)
		return out
		
	def meancolors(self):
		cell_colors = {}
		for cellnumber, set in self.filled_sets.items():
			L = 0
			a = 0
			b = 0
			for pixel in set:
				L += self.pixel_map[pixel][0]
				a += self.pixel_map[pixel][1]
				b += self.pixel_map[pixel][2]
			L/=len(set)
			a/=len(set)
			b/=len(set)
			cell_colors[cellnumber] = (L, a, b)
		return cell_colors
	
	def maxneighborcell(self, cellnum):
		neighbors = []
		for pixel in self.filled_sets[cellnum]:
			for item in [(pixel[0]+1, pixel[1]), (pixel[0]-1,pixel[1]), (pixel[0], pixel[1]+1), (pixel[0], pixel[1]-1)]:
				try:
					if(item not in self.filled_sets[cellnum] and (len(self.filled_sets[self.pixel_cells_map[item]]) < SMALL_CELL_TOLERANCE or len(neighbors) == 0)):
						neighbors.append(self.pixel_cells_map[item])
				except KeyError:
					continue
		return neighbors
		
def lm_quantizer(image,k):
    im_shape = image.shape
    n_rows = im_shape[0]
    n_cols = im_shape[1]
    kmeans = KMeans(n_clusters = k)
    if len(im_shape) == 2:
        pixel_vals = np.array([[image[row,col]] for row in range(n_rows) for col in range(n_cols)])
    else:
        pixel_vals = np.array([image[row,col] for row in range(n_rows) for col in range(n_cols)])
    color_labels = kmeans.fit_predict(pixel_vals)
    q_image = np.zeros(im_shape).astype(np.uint8)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    for i,label in enumerate(color_labels):
        q_image[int(i/n_cols),i % n_cols] = colors[label]
    return q_image, colors

def rgb2xyz(rgb):
	r,g,b = rgb
	r/=255
	g/=255
	b/=255
	if(r>0.04045):
		r=((r+0.055)/1.055)**2.4
	else:
		r/=12.92
	if(g>0.04045):
		g=((g+0.055)/1.055)**2.4
	else:
		g/=12.92
	if(b>0.04045):
		b=((b+0.055)/1.055)**2.4
	else:
		b/=12.92
	r*=100
	g*=100
	b*=100
	x=r*0.4124+g*0.3576+b*0.1805
	y=r*0.2126+g*0.7152+b*0.0722
	z=r*0.0193+g*0.1192+b*0.9505
	return (x,y,z)

def xyz2lab(xyz):
	x,y,z=xyz
	x/=95.047
	y/=100
	z/=108.883
	if(x>0.008856):
		x=x**(1/3)
	else:
		x=7.787*x+16/116
	if(y>0.008856):
		y=y**(1/3)
	else:
		y=7.787*y+16/116
	if(z>0.008856):
		z=z**(1/3)
	else:
		z=7.787*z + 16/116
	L=116*y-16
	a=500*(x-y)
	b=200*(y-z)
	return (L,a,b)

def rgb2lab(rgb):
	return xyz2lab(rgb2xyz(rgb))
	
def lab2xyz(lab):
	L,a,b=lab
	y=(L+16)/116
	x=a/500+y
	z=y-b/200
	if(y**3>0.008856):
		y=y**3
	else:
		y=(y-16/116)/7.787
	if(x**3>0.008856):
		x=x**3
	else:
		x=(x-16/116)/7.787
	if(z**3>0.008856):
		z=z**3
	else:
		z=(z-16/116)/7.787
	x*=95.047
	y*=100
	z*=108.883
	return(x,y,z)
	
def xyz2rgb(xyz):
	x,y,z=xyz
	x/=100
	y/=100
	z/=100
	r=x*3.2406+y*-1.5372+z*-0.4986
	g=x*-0.9689+y*1.8758+z*0.0415
	b=x*0.0557+y*-0.2040+z*1.0570
	if(r>0.0031308):
		r=1.055*(r**(1/2.4))-0.055
	else:
		r=12.92*r
	if(g>0.0031308):
		g=1.055*(g**(1/2.4))-0.055
	else:
		g=12.92*g
	if(b>0.0031308):
		b=1.055*(b**(1/2.4))-0.055
	else:
		b=12.92*b
	r*=255
	g*=255
	b*=255
	return(r,g,b)

def lab2rgb(lab):
	rgb=xyz2rgb(lab2xyz(lab))
	return tuple([int(round(x))for x in rgb])
	
def hex_to_rgb(num):
    h = str(num)
    return int(h[0:4], 16), int(('0x' + h[4:6]), 16), int(('0x' + h[6:8]), 16)
	
def rgb_to_hex(num):
    h = str(num)
    return int(h[0:4], 16), int(('0x' + h[4:6]), 16), int(('0x' + h[6:8]), 16)

def colordif(color1, color2):
	return (color1[0]-color2[0])**2 + (color1[1]-color2[1])**2 + (color1[2]-color2[2])**2


filename = input("What image would you like to use? (include the file type)\n")
total_cells = int(input("How many cells would you like to include in the final image?\n"))
total_colors = int(input("How many colors would you like to use in the image?\n"))
input = ImageData(IMAGE_FILEPATH + filename)
print("Image loaded successfully")
print("Assigning color cells...")
cells = 0
while(len(input.unassigned_pixels) != 0):
	input.floodfill(input.unassigned_pixels.pop(), cells)
	cells += 1
print("Flood Fill completed with " + str(len(input.filled_sets)) + " color cells created")
print("Finding small cells...")
input.smallcells(SMALL_CELL_TOLERANCE)
print("Small Cells removed with " + str(len(input.filled_sets)) + " color cells remaining")
input.mergesimilar(PASS_TWO_COLOR_TOLERANCE)
print("Similar Cells merged with " + str(len(input.filled_sets)) + " color cells remaining")
tol = 2*SMALL_CELL_TOLERANCE
color_tol = PASS_TWO_COLOR_TOLERANCE
while(len(input.filled_sets) > total_cells):
	print("Finding small cells...")
	input.smallcells(tol)
	print("Small Cells removed with " + str(len(input.filled_sets)) + " color cells remaining")
	input.mergesimilar(color_tol)
	print("Similar Cells merged with " + str(len(input.filled_sets)) + " color cells remaining")
	tol *= 2
	color_tol *= 2
print("Small Cells removed with " + str(len(input.filled_sets)) + " color cells remaining")
input.outputimage(filename)
print("Grouped Image Completed and Saved")
grouped = io.imread("C:/Users/trgre/Desktop/Programming/Python/PaintByNumber/grouped_" + filename)
painted, palette = lm_quantizer(grouped, total_colors)
print(palette)
painted_img = Image.fromarray(painted)
print("Similar Colors Quantized to a total of " + str(total_colors) + " colors")
painted_img.save("C:/Users/trgre/Desktop/Programming/Python/PaintByNumber/painted_" + filename)
print("Painted Image Completed and Saved")
line_img = input.bordercolored(filename)
print("Image Outline Drawn")
input.addnumbers(filename, line_img, painted_img, palette)
print("Color Codes added to outline and saved")



