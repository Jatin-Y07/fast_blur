def box_blur(pixels, width, height, channels, radius):
    output = bytearray(len(pixels))
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                total = 0
                count = 0
                for ky in range(-radius, radius + 1):
                    for kx in range(-radius, radius + 1):
                        nx, ny = x + kx, y + ky
                        if 0 <= nx < width and 0 <= ny < height:
                            total += pixels[(ny * width + nx) * channels + c]
                            count += 1
                output[(y * width + x) * channels + c] = total // count
    return bytes(output)
