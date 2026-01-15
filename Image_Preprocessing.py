# 1. Load and preprocess image
# Load image from image_path
# Convert image to RGB format

from PIL import Image

image_path = 'mario-day-mario-headshot.jpg'  # <-- update the name with the image you want
image = Image.open(image_path)

image = image.convert('RGB')

width, height = image.size
print(f"Image size: {width}x{height}") # the image size of the high res goldengate bridge was 1000 x 750 so 750000 pixels
image.show()  # Opens the image in default viewer


# 2. Choosing Cube to Pixel Ratio
# Generally, having one face of cube (9 stickers, each sticker representing 10 pixels) 
# representing 30 pixels is reasonable to balance the detail and cube count

original_width = width # 1000 for GGB
original_height = height # 750 for GGB

cube_face_size = 30

# Calculate how many cubes fit across and down
cubes_wide = original_width // cube_face_size
cubes_high = original_height // cube_face_size

mosaic_pixel_width  = cubes_wide * cube_face_size # the dimension in pixels that perfectly fits your Rubik’s cube mosaic. (integer division)
mosaic_pixel_height = cubes_high * cube_face_size

# so in the GGB example, our rubik's cube is going to draw 990 x 750

resized_image = image.resize((mosaic_pixel_width, mosaic_pixel_height))
resized_image

# resized_image.show()









# 3. Quantize image colors to Rubik's cube palette

# Define RGB values for W, Y, R, O, G, B
# For each pixel in resized image:
    #Assign pixel to closest color in palette (using Euclidean RGB distance)



# 4. Divide image into 3x3 blocks (one per cube)

#Initialize empty list cube_patterns = []

#For row in 0 to mosaic_height in steps of 3:
    #For col in 0 to mosaic_width in steps of 3:
        #Extract 3x3 block of pixels
        #Convert pixel RGBs to corresponding cube color codes (W, Y, etc.)
        #Save 3x3 block as cube_pattern
        #Append to cube_patterns


# 5. Output cube patterns

#Display or save each cube's 3x3 pattern in order
#(Optional) Visualize full mosaic as color-coded grid
#(Optional) Export instructions in CSV, JSON, or image format









