import numpy as np
import matplotlib.pyplot as plt

R = 1.0

SIZE = 0.008

POINTS = 1200

I0 = 1.0

MODE = "quasimonochromatic"

CENTER_WAVELENGTH = 550

SPECTRUM_WIDTH = 80

x = np.linspace(-SIZE, SIZE, POINTS)
y = np.linspace(-SIZE, SIZE, POINTS)

X, Y = np.meshgrid(x, y)

r = np.sqrt(X**2 + Y**2)

def wavelength_to_rgb(wavelength_nm):

    gamma = 0.8

    if 380 <= wavelength_nm <= 440:

        attenuation = 0.3 + 0.7 * (wavelength_nm - 380) / (440 - 380)

        R = ((-(wavelength_nm - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma

    elif 440 <= wavelength_nm <= 490:

        R = 0.0
        G = ((wavelength_nm - 440) / (490 - 440)) ** gamma
        B = 1.0

    elif 490 <= wavelength_nm <= 510:

        R = 0.0
        G = 1.0
        B = (-(wavelength_nm - 510) / (510 - 490)) ** gamma

    elif 510 <= wavelength_nm <= 580:

        R = ((wavelength_nm - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0

    elif 580 <= wavelength_nm <= 645:

        R = 1.0
        G = (-(wavelength_nm - 645) / (645 - 580)) ** gamma
        B = 0.0

    elif 645 <= wavelength_nm <= 780:

        attenuation = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 645)

        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0

    else:

        R = G = B = 0.0

    return np.array([R, G, B])


if MODE == "monochromatic":

    wavelengths = [CENTER_WAVELENGTH]

    weights = [1.0]

else:

    wavelengths = np.linspace(
        CENTER_WAVELENGTH - SPECTRUM_WIDTH / 2,
        CENTER_WAVELENGTH + SPECTRUM_WIDTH / 2,
        15
    )

    sigma = SPECTRUM_WIDTH / 2.355

    weights = np.exp(
        -(wavelengths - CENTER_WAVELENGTH) ** 2 / (2 * sigma**2)
    )

    weights /= weights.sum()


image = np.zeros((POINTS, POINTS, 3))

for wavelength_nm, weight in zip(wavelengths, weights):

    wavelength = wavelength_nm * 1e-9

    t = r**2 / (2 * R)

    delta = 2 * t

    phi = 2 * np.pi * delta / wavelength

    intensity = I0 * np.sin(phi / 2) ** 2

    rgb = wavelength_to_rgb(wavelength_nm)

    for i in range(3):

        image[:, :, i] += weight * intensity * rgb[i]


image /= image.max()


plt.figure(figsize=(9, 9))

plt.imshow(
    image,
    extent=[
        -SIZE * 1000,
        SIZE * 1000,
        -SIZE * 1000,
        SIZE * 1000
    ]
)

plt.title("Кольца Ньютона")

plt.xlabel("x (мм)")
plt.ylabel("y (мм)")

plt.show()


r_line = np.linspace(0, SIZE, POINTS)

intensity_total = np.zeros_like(r_line)

for wavelength_nm, weight in zip(wavelengths, weights):

    wavelength = wavelength_nm * 1e-9

    t_line = r_line**2 / (2 * R)

    delta_line = 2 * t_line

    phi_line = 2 * np.pi * delta_line / wavelength

    intensity_line = I0 * np.sin(phi_line / 2) ** 2

    intensity_total += weight * intensity_line


intensity_total /= intensity_total.max()


plt.figure(figsize=(10, 5))

plt.plot(r_line * 1000, intensity_total)

plt.title("Зависимость интенсивности от радиальной координаты")

plt.xlabel("Радиус r (мм)")
plt.ylabel("Интенсивность")

plt.grid()

plt.show()