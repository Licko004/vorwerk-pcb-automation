# import time
# import board
# import neopixel

# # Configure the setup
# PIXEL_PIN = board.D10       # pin that the NeoPixel ring is connected to
# NUM_PIXELS = 24              # your ring has 24 LEDs
# ORDER = neopixel.GRB         # most 24-LED NeoPixel rings use GRB, not RGB
# BRIGHTNESS = 1.0             # full brightness

# pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

# pixels.fill((255, 255, 255))
# pixels.show()

# GRBW 
import time
import board
import neopixel

# Configure the setup
PIXEL_PIN = board.D18       # pin that the NeoPixel ring is connected to
NUM_PIXELS = 24              # your ring has 24 LEDs
ORDER = neopixel.GRBW        # RGBW ring
BRIGHTNESS = 1.0             # full brightness
print("before enopixel.line")
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, pixel_order=ORDER)

pixels.fill((0, 0, 0, 255))   # R, G, B, W
pixels.show()
print("after program")

time.sleep(10)

pixels.fill((0, 0, 0, 0))   # R, G, B, W
pixels.show()