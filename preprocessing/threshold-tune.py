import cv2

def on_trackbar(threshold_value):
    global img
    thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_TOZERO)[1]
    cv2.imshow('Thresholded Image', thresholded_img)

### SET THIS VARIABLE ###
img_file = "/media/nick/C8EB-647B/Data/processed/2_chs/frames/Exp_1/C/frame_00000.png"
#########################


img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

# Initialize threshold value
initial_threshold_value = 128

# Create a window
cv2.namedWindow('Threshold Slider')

# Create a trackbar
cv2.createTrackbar('Threshold', 'Threshold Slider', initial_threshold_value, 255, on_trackbar)

# Show the initial thresholded image
on_trackbar(initial_threshold_value)

# Wait until a key is pressed
cv2.waitKey(0)

# Print the final threshold value
final_threshold_value = cv2.getTrackbarPos('Threshold', 'Threshold Slider')
print("Final Threshold Value:", final_threshold_value)

cv2.destroyAllWindows()
