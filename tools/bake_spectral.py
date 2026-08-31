#!/usr/bin/env python3
"""Bakes the spectral tables the renderer needs into assets/bin/spectral.bin.

Three things go in the blob:

  * a Jakob & Hanika 2019 coefficient LUT, which turns an authored sRGB colour into a smooth
    reflectance spectrum,
  * the CIE 1931 2 degree colour matching functions, so the shader and the C side resolve
    spectra to XYZ through exactly the same observer,
  * the CIE D-series daylight basis S0/S1/S2, which the sky uses to turn Preetham's xyY into
    a spectrum, and which also defines D65 -- the illuminant the LUT is fitted under.

Unlike tools/bake_assets.py this one uses numpy. The fit is 786432 independent three
parameter least squares problems and pure Python would take hours; the asset baker's
stdlib-only rule is about not owing a mesh parser to anyone, and does not apply here.

    tools/bake_spectral.py            # no-op when the blob is newer than this script
    tools/bake_spectral.py --force
    tools/bake_spectral.py --resolution 32 --verbose

The measured data below is not hand transcribed: the colour matching functions come from the
CVRL tabulation of CIE 1931 2 degree at 1nm, and the daylight basis from CIE 15. Both were
checked on the way in -- the CMF table integrates to 106.8569 against the published
106.856895, and the basis reconstructs to exactly 100 at 560nm for any chromaticity, which is
the normalisation CIE defines it with.
"""

import argparse
import os
import struct
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("bake_spectral: numpy is required -- install it with 'pacman -S python-numpy'")

# ---------------------------------------------------------------------------
# format
#
# Mirrored by pt_spectral_header_t in src/spectral.c, which _Static_asserts the 128 bytes and
# rejects any version but this one by name.
# ---------------------------------------------------------------------------

MAGIC = b"PTSPEC01"
VERSION = 2  # bumped when the payload gained the leading constants block
HEADER_SIZE = 128
NAME_SIZE = 48

# The sampling grid every spectrum in the blob is resampled onto, and the range the shader
# draws its wavelengths from. 1nm over the full CIE range: the tables are small enough that
# there is nothing to gain from coarser, and matching the published domain exactly means the
# integrals below can be checked against published constants.
LAMBDA_MIN = 360.0
LAMBDA_MAX = 830.0
LAMBDA_COUNT = 471

DEFAULT_RESOLUTION = 64

# sRGB primaries (IEC 61966-2-1). The white point is deliberately *not* taken from here -- see
# white_point() for why it is computed from the reconstructed D65 spectrum instead.
SRGB_PRIMARIES = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
D65_CHROMATICITY = (0.3127, 0.3290)

# ---------------------------------------------------------------------------
# measured data
# ---------------------------------------------------------------------------

# CIE 1931 2 degree colour matching functions, x/y/z interleaved, 360..830nm at 1nm.
CIE_XYZ_1NM = (
    0.0001299, 3.917e-06, 0.0006061, 0.000145847, 4.39358e-06, 0.000680879, 0.000163802,
    4.9296e-06, 0.000765146, 0.000184004, 5.53214e-06, 0.000860012, 0.00020669, 6.20824e-06,
    0.000966593, 0.0002321, 6.965e-06, 0.001086, 0.000260728, 7.81322e-06, 0.00122059,
    0.000293075, 8.76734e-06, 0.00137273, 0.000329388, 9.83984e-06, 0.00154358, 0.000369914,
    1.10432e-05, 0.00173429, 0.0004149, 1.239e-05, 0.001946, 0.000464159, 1.38864e-05,
    0.00217778, 0.000518986, 1.55573e-05, 0.00243581, 0.000581854, 1.7443e-05, 0.00273195,
    0.000655235, 1.95838e-05, 0.00307806, 0.0007416, 2.202e-05, 0.003486, 0.00084503,
    2.48396e-05, 0.00397523, 0.000964527, 2.80413e-05, 0.00454088, 0.00109495, 3.1531e-05,
    0.00515832, 0.00123115, 3.52152e-05, 0.00580291, 0.001368, 3.9e-05, 0.00645, 0.00150205,
    4.28264e-05, 0.00708322, 0.00164233, 4.69146e-05, 0.00774549, 0.00180238, 5.15896e-05,
    0.00850115, 0.00199576, 5.71764e-05, 0.00941454, 0.002236, 6.4e-05, 0.01055, 0.00253538,
    7.23442e-05, 0.0119658, 0.0028926, 8.22122e-05, 0.0136559, 0.00330083, 9.35082e-05,
    0.0155881, 0.00375324, 0.000106136, 0.0177302, 0.004243, 0.00012, 0.02005, 0.00476239,
    0.000134984, 0.0225114, 0.00533005, 0.000151492, 0.0252029, 0.00597871, 0.000170208,
    0.0282797, 0.00674112, 0.000191816, 0.031897, 0.00765, 0.000217, 0.03621, 0.00875137,
    0.000246907, 0.0414377, 0.0100289, 0.00028124, 0.0475037, 0.0114217, 0.00031852, 0.0541199,
    0.012869, 0.000357267, 0.060998, 0.01431, 0.000396, 0.06785, 0.0157044, 0.000433715,
    0.0744863, 0.0171474, 0.000473024, 0.0813616, 0.0187812, 0.000517876, 0.0891536, 0.020748,
    0.000572219, 0.0985405, 0.02319, 0.00064, 0.1102, 0.0262074, 0.00072456, 0.124613,
    0.0297825, 0.0008255, 0.141702, 0.0338809, 0.00094116, 0.161303, 0.0384682, 0.00106988,
    0.183257, 0.04351, 0.00121, 0.2074, 0.0489956, 0.00136209, 0.233692, 0.0550226, 0.00153075,
    0.262611, 0.0617188, 0.00172037, 0.294775, 0.069212, 0.00193532, 0.330798, 0.07763, 0.00218,
    0.3713, 0.0869581, 0.0024548, 0.416209, 0.0971767, 0.002764, 0.465464, 0.108406, 0.0031178,
    0.519695, 0.120767, 0.0035264, 0.57953, 0.13438, 0.004, 0.6456, 0.149358, 0.00454624,
    0.718484, 0.165396, 0.00515932, 0.796713, 0.181983, 0.00582928, 0.877846, 0.198611,
    0.00654616, 0.959439, 0.21477, 0.0073, 1.03905, 0.230187, 0.00808651, 1.11537, 0.24488,
    0.00890872, 1.1885, 0.258777, 0.00976768, 1.25812, 0.271808, 0.0106644, 1.32393, 0.2839,
    0.0116, 1.3856, 0.294944, 0.0125732, 1.44264, 0.304897, 0.0135827, 1.4948, 0.313787,
    0.0146297, 1.54219, 0.321645, 0.0157151, 1.58488, 0.3285, 0.01684, 1.62296, 0.334351,
    0.0180074, 1.6564, 0.33921, 0.0192145, 1.6853, 0.343121, 0.0204539, 1.70987, 0.34613,
    0.0217182, 1.73038, 0.34828, 0.023, 1.74706, 0.3496, 0.0242946, 1.76004, 0.350147,
    0.0256102, 1.76962, 0.350013, 0.0269586, 1.77626, 0.349287, 0.0283513, 1.78043, 0.34806,
    0.0298, 1.7826, 0.346373, 0.0313108, 1.78297, 0.344262, 0.0328837, 1.7817, 0.341809,
    0.0345211, 1.7792, 0.339094, 0.0362257, 1.77587, 0.3362, 0.038, 1.77211, 0.333198,
    0.0398467, 1.76826, 0.330041, 0.041768, 1.76404, 0.326636, 0.043766, 1.75894, 0.322887,
    0.0458427, 1.75247, 0.3187, 0.048, 1.7441, 0.314025, 0.0502437, 1.73356, 0.308884, 0.052573,
    1.72086, 0.30329, 0.0549806, 1.70594, 0.297258, 0.0574587, 1.68874, 0.2908, 0.06, 1.6692,
    0.28397, 0.062602, 1.64753, 0.276721, 0.0652775, 1.62341, 0.268918, 0.0680421, 1.59602,
    0.260423, 0.0709111, 1.56453, 0.2511, 0.0739, 1.5281, 0.240847, 0.077016, 1.48611, 0.229851,
    0.0802664, 1.43952, 0.218407, 0.0836668, 1.38988, 0.206812, 0.0872328, 1.33874, 0.19536,
    0.09098, 1.28764, 0.184214, 0.0949176, 1.23742, 0.173327, 0.0990458, 1.18782, 0.162688,
    0.103367, 1.13876, 0.152283, 0.107885, 1.09015, 0.1421, 0.1126, 1.0419, 0.132179, 0.117532,
    0.994198, 0.12257, 0.122674, 0.947347, 0.113275, 0.127993, 0.901453, 0.104298, 0.133453,
    0.856619, 0.09564, 0.13902, 0.81295, 0.0872996, 0.144676, 0.770517, 0.079308, 0.150469,
    0.729445, 0.0717178, 0.156462, 0.689914, 0.064581, 0.162718, 0.652105, 0.05795, 0.1693,
    0.6162, 0.0518621, 0.176243, 0.582329, 0.0462815, 0.183558, 0.550416, 0.0411509, 0.191274,
    0.520338, 0.0364128, 0.199418, 0.491967, 0.03201, 0.20802, 0.46518, 0.0279172, 0.21712,
    0.439925, 0.0241444, 0.226735, 0.416184, 0.020687, 0.236857, 0.393882, 0.0175404, 0.247481,
    0.372946, 0.0147, 0.2586, 0.3533, 0.0121618, 0.270185, 0.334858, 0.00991996, 0.282294,
    0.317552, 0.00796724, 0.29505, 0.301337, 0.00629635, 0.308578, 0.286169, 0.0049, 0.323,
    0.272, 0.00377717, 0.338402, 0.258817, 0.00294532, 0.354686, 0.246484, 0.00242488, 0.371699,
    0.234772, 0.00223629, 0.389288, 0.223453, 0.0024, 0.4073, 0.2123, 0.00292552, 0.42563,
    0.201169, 0.00383656, 0.44431, 0.19012, 0.00517484, 0.463394, 0.179225, 0.00698208, 0.48294,
    0.168561, 0.0093, 0.503, 0.1582, 0.0121495, 0.523569, 0.148138, 0.0155359, 0.544512,
    0.138376, 0.0194775, 0.56569, 0.128994, 0.0239928, 0.586965, 0.120075, 0.0291, 0.6082,
    0.1117, 0.0348149, 0.629346, 0.103905, 0.0411202, 0.650307, 0.0966675, 0.047985, 0.670875,
    0.0899827, 0.0553786, 0.690842, 0.0838453, 0.06327, 0.71, 0.07825, 0.071635, 0.728185,
    0.073209, 0.0804622, 0.745464, 0.0686782, 0.08974, 0.761969, 0.0645678, 0.0994565, 0.777837,
    0.0607883, 0.1096, 0.7932, 0.05725, 0.120167, 0.80811, 0.0539043, 0.131114, 0.822496,
    0.0507466, 0.142368, 0.836307, 0.0477528, 0.153854, 0.849492, 0.0448986, 0.1655, 0.862,
    0.04216, 0.177257, 0.873811, 0.0395073, 0.18914, 0.884962, 0.0369356, 0.201169, 0.895494,
    0.0344584, 0.213366, 0.905443, 0.0320887, 0.22575, 0.91485, 0.02984, 0.238321, 0.923735,
    0.0277118, 0.251067, 0.932092, 0.0256944, 0.263992, 0.939923, 0.0237872, 0.277102, 0.947225,
    0.0219892, 0.2904, 0.954, 0.0203, 0.303891, 0.960256, 0.018718, 0.317573, 0.966007,
    0.0172404, 0.331438, 0.971261, 0.0158636, 0.345483, 0.976023, 0.0145846, 0.3597, 0.9803,
    0.0134, 0.374084, 0.984092, 0.0123072, 0.38864, 0.987418, 0.0113019, 0.403378, 0.990313,
    0.0103779, 0.418312, 0.992812, 0.00952931, 0.43345, 0.99495, 0.00875, 0.448795, 0.996711,
    0.0080352, 0.464336, 0.998098, 0.0073816, 0.480064, 0.999112, 0.0067854, 0.495971, 0.999748,
    0.0062428, 0.51205, 1, 0.00575, 0.528296, 0.999857, 0.0053036, 0.544692, 0.999305,
    0.0048998, 0.561209, 0.998325, 0.0045342, 0.577821, 0.996899, 0.0042024, 0.5945, 0.995,
    0.0039, 0.611221, 0.9926, 0.0036232, 0.627976, 0.989743, 0.0033706, 0.64476, 0.986444,
    0.0031414, 0.66157, 0.982724, 0.0029348, 0.6784, 0.9786, 0.00275, 0.695239, 0.974084,
    0.0025852, 0.712059, 0.969171, 0.0024386, 0.728828, 0.963857, 0.0023094, 0.745519, 0.958135,
    0.0021968, 0.7621, 0.952, 0.0021, 0.778543, 0.94545, 0.00201773, 0.794826, 0.938499,
    0.0019482, 0.810926, 0.931163, 0.0018898, 0.826825, 0.923458, 0.00184093, 0.8425, 0.9154,
    0.0018, 0.857932, 0.907006, 0.00176627, 0.873082, 0.898277, 0.0017378, 0.887894, 0.889205,
    0.0017112, 0.902318, 0.879782, 0.00168307, 0.9163, 0.87, 0.00165, 0.9298, 0.859861,
    0.00161013, 0.942798, 0.849392, 0.0015644, 0.955278, 0.838622, 0.0015136, 0.967218,
    0.827581, 0.00145853, 0.9786, 0.8163, 0.0014, 0.989386, 0.804795, 0.00133667, 0.999549,
    0.793082, 0.00127, 1.00909, 0.781192, 0.001205, 1.01801, 0.769155, 0.00114667, 1.0263,
    0.757, 0.0011, 1.03398, 0.744754, 0.0010688, 1.04099, 0.732422, 0.0010494, 1.04719,
    0.720004, 0.0010356, 1.05247, 0.707496, 0.0010212, 1.0567, 0.6949, 0.001, 1.05979, 0.682219,
    0.00096864, 1.0618, 0.669472, 0.00092992, 1.06281, 0.656674, 0.00088688, 1.06291, 0.643845,
    0.00084256, 1.0622, 0.631, 0.0008, 1.06074, 0.618155, 0.00076096, 1.05844, 0.605314,
    0.00072368, 1.05522, 0.592476, 0.00068592, 1.05098, 0.579638, 0.00064544, 1.0456, 0.5668,
    0.0006, 1.03904, 0.553961, 0.000547867, 1.03136, 0.541137, 0.0004916, 1.02267, 0.528353,
    0.0004354, 1.01305, 0.515632, 0.000383467, 1.0026, 0.503, 0.00034, 0.991367, 0.490469,
    0.000307253, 0.979331, 0.47803, 0.00028316, 0.966492, 0.465678, 0.00026544, 0.952848,
    0.453403, 0.000251813, 0.9384, 0.4412, 0.00024, 0.923194, 0.42908, 0.000229547, 0.907244,
    0.417036, 0.00022064, 0.890502, 0.405032, 0.00021196, 0.87292, 0.393032, 0.000202187,
    0.85445, 0.381, 0.00019, 0.835084, 0.368918, 0.000174213, 0.814946, 0.356827, 0.00015564,
    0.794186, 0.344777, 0.00013596, 0.772954, 0.332818, 0.000116853, 0.7514, 0.321, 0.0001,
    0.729584, 0.309338, 8.61333e-05, 0.707589, 0.29785, 7.46e-05, 0.685602, 0.286594, 6.5e-05,
    0.66381, 0.275624, 5.69333e-05, 0.6424, 0.265, 5e-05, 0.621515, 0.254763, 4.416e-05,
    0.601114, 0.24489, 3.948e-05, 0.581105, 0.235334, 3.572e-05, 0.561398, 0.226053, 3.264e-05,
    0.5419, 0.217, 3e-05, 0.522599, 0.208162, 2.76533e-05, 0.503546, 0.199549, 2.556e-05,
    0.484744, 0.191155, 2.364e-05, 0.466194, 0.182974, 2.18133e-05, 0.4479, 0.175, 2e-05,
    0.429861, 0.167223, 1.81333e-05, 0.412098, 0.159646, 1.62e-05, 0.394644, 0.152278, 1.42e-05,
    0.377533, 0.145126, 1.21333e-05, 0.3608, 0.1382, 1e-05, 0.344456, 0.1315, 7.73333e-06,
    0.328517, 0.125025, 5.4e-06, 0.313019, 0.118779, 3.2e-06, 0.298001, 0.112769, 1.33333e-06,
    0.2835, 0.107, 0, 0.269545, 0.101476, 0, 0.256118, 0.0961886, 0, 0.24319, 0.091123, 0,
    0.230727, 0.0862649, 0, 0.2187, 0.0816, 0, 0.207097, 0.0771206, 0, 0.195923, 0.0728255, 0,
    0.185171, 0.0687101, 0, 0.174832, 0.0647698, 0, 0.1649, 0.061, 0, 0.155367, 0.0573962, 0,
    0.14623, 0.053955, 0, 0.13749, 0.0506738, 0, 0.129147, 0.0475496, 0, 0.1212, 0.04458, 0,
    0.11364, 0.0417587, 0, 0.106465, 0.039085, 0, 0.0996904, 0.0365638, 0, 0.0933306, 0.0342005,
    0, 0.0874, 0.032, 0, 0.081901, 0.0299626, 0, 0.0768043, 0.0280766, 0, 0.0720771, 0.0263294,
    0, 0.0676866, 0.024708, 0, 0.0636, 0.0232, 0, 0.0598069, 0.0218008, 0, 0.0562822, 0.0205011,
    0, 0.052971, 0.0192811, 0, 0.0498186, 0.0181207, 0, 0.04677, 0.017, 0, 0.043784, 0.0159038,
    0, 0.0408754, 0.0148372, 0, 0.0380726, 0.0138107, 0, 0.0354046, 0.0128348, 0, 0.0329,
    0.01192, 0, 0.0305642, 0.0110683, 0, 0.0283806, 0.0102734, 0, 0.0263448, 0.00953331, 0,
    0.0244527, 0.00884616, 0, 0.0227, 0.00821, 0, 0.0210843, 0.00762378, 0, 0.0195999,
    0.00708542, 0, 0.0182373, 0.00659148, 0, 0.0169872, 0.00613848, 0, 0.01584, 0.005723, 0,
    0.0147906, 0.00534306, 0, 0.0138313, 0.0049958, 0, 0.0129487, 0.0046764, 0, 0.0121292,
    0.00438007, 0, 0.0113592, 0.004102, 0, 0.0106293, 0.00383845, 0, 0.00993885, 0.0035891, 0,
    0.00928842, 0.00335422, 0, 0.00867885, 0.00313409, 0, 0.00811092, 0.002929, 0, 0.00758239,
    0.00273814, 0, 0.00708875, 0.00255988, 0, 0.00662731, 0.00239324, 0, 0.00619541, 0.00223727,
    0, 0.00579035, 0.002091, 0, 0.00540983, 0.00195359, 0, 0.00505258, 0.00182458, 0,
    0.00471751, 0.00170358, 0, 0.00440351, 0.00159019, 0, 0.00410946, 0.001484, 0, 0.00383391,
    0.0013845, 0, 0.00357575, 0.00129127, 0, 0.00333434, 0.00120409, 0, 0.00310908, 0.00112274,
    0, 0.00289933, 0.001047, 0, 0.00270435, 0.00097659, 0, 0.00252302, 0.000911109, 0,
    0.00235417, 0.000850133, 0, 0.00219662, 0.000793238, 0, 0.00204919, 0.00074, 0, 0.00191096,
    0.000690083, 0, 0.00178144, 0.00064331, 0, 0.00166011, 0.000599496, 0, 0.00154646,
    0.000558455, 0, 0.00143997, 0.00052, 0, 0.00134004, 0.000483914, 0, 0.00124628, 0.000450053,
    0, 0.00115847, 0.000418345, 0, 0.00107643, 0.000388718, 0, 0.000999949, 0.0003611, 0,
    0.000928736, 0.000335383, 0, 0.000862433, 0.00031144, 0, 0.00080075, 0.000289166, 0,
    0.000743396, 0.000268454, 0, 0.000690079, 0.0002492, 0, 0.000640516, 0.000231302, 0,
    0.000594502, 0.000214686, 0, 0.000551865, 0.000199288, 0, 0.000512429, 0.000185048, 0,
    0.000476021, 0.0001719, 0, 0.000442454, 0.000159778, 0, 0.000411512, 0.000148604, 0,
    0.000382981, 0.000138302, 0, 0.000356649, 0.000128793, 0, 0.000332301, 0.00012, 0,
    0.000309759, 0.000111859, 0, 0.000288887, 0.000104322, 0, 0.000269539, 9.73356e-05, 0,
    0.000251568, 9.08459e-05, 0, 0.000234826, 8.48e-05, 0, 0.000219171, 7.91467e-05, 0,
    0.000204526, 7.3858e-05, 0, 0.00019084, 6.8916e-05, 0, 0.000178065, 6.43027e-05, 0,
    0.000166151, 6e-05, 0, 0.000155024, 5.59819e-05, 0, 0.000144622, 5.22256e-05, 0, 0.00013491,
    4.87184e-05, 0, 0.000125852, 4.54475e-05, 0, 0.000117413, 4.24e-05, 0, 0.000109552,
    3.9561e-05, 0, 0.000102224, 3.69151e-05, 0, 9.53945e-05, 3.44487e-05, 0, 8.90239e-05,
    3.21482e-05, 0, 8.30753e-05, 3e-05, 0, 7.75127e-05, 2.79913e-05, 0, 7.2313e-05, 2.61136e-05,
    0, 6.74578e-05, 2.43602e-05, 0, 6.29284e-05, 2.27246e-05, 0, 5.87065e-05, 2.12e-05, 0,
    5.47703e-05, 1.97786e-05, 0, 5.10992e-05, 1.84529e-05, 0, 4.76765e-05, 1.72169e-05, 0,
    4.44857e-05, 1.60646e-05, 0, 4.15099e-05, 1.499e-05, 0, 3.87332e-05, 1.39873e-05, 0,
    3.6142e-05, 1.30516e-05, 0, 3.37235e-05, 1.21782e-05, 0, 3.14649e-05, 1.13625e-05, 0,
    2.93533e-05, 1.06e-05, 0, 2.73757e-05, 9.88588e-06, 0, 2.55243e-05, 9.2173e-06, 0,
    2.37938e-05, 8.59236e-06, 0, 2.21787e-05, 8.00913e-06, 0, 2.06738e-05, 7.4657e-06, 0,
    1.92723e-05, 6.95957e-06, 0, 1.79664e-05, 6.488e-06, 0, 1.67499e-05, 6.0487e-06, 0,
    1.56165e-05, 5.6394e-06, 0, 1.45598e-05, 5.2578e-06, 0, 1.35739e-05, 4.90177e-06, 0,
    1.26544e-05, 4.56972e-06, 0, 1.17972e-05, 4.26019e-06, 0, 1.09984e-05, 3.97174e-06, 0,
    1.0254e-05, 3.7029e-06, 0, 9.55965e-06, 3.45216e-06, 0, 8.91204e-06, 3.2183e-06, 0,
    8.30836e-06, 3.0003e-06, 0, 7.74577e-06, 2.79714e-06, 0, 7.22146e-06, 2.6078e-06, 0,
    6.73248e-06, 2.43122e-06, 0, 6.27642e-06, 2.26653e-06, 0, 5.8513e-06, 2.11301e-06, 0,
    5.45512e-06, 1.96994e-06, 0, 5.08587e-06, 1.8366e-06, 0, 4.74147e-06, 1.71223e-06, 0,
    4.42024e-06, 1.59623e-06, 0, 4.12078e-06, 1.48809e-06, 0, 3.84172e-06, 1.38731e-06, 0,
    3.58165e-06, 1.2934e-06, 0, 3.33913e-06, 1.20582e-06, 0, 3.11295e-06, 1.12414e-06, 0,
    2.90212e-06, 1.04801e-06, 0, 2.70565e-06, 9.77058e-07, 0, 2.52253e-06, 9.1093e-07, 0,
    2.35173e-06, 8.49251e-07, 0, 2.19242e-06, 7.91721e-07, 0, 2.0439e-06, 7.3809e-07, 0,
    1.9055e-06, 6.8811e-07, 0, 1.77651e-06, 6.4153e-07, 0, 1.65621e-06, 5.9809e-07, 0,
    1.54402e-06, 5.57575e-07, 0, 1.43944e-06, 5.19808e-07, 0, 1.34198e-06, 4.84612e-07, 0,
    1.25114e-06, 4.5181e-07, 0
)

# CIE D-series daylight basis, S0/S1/S2 interleaved, 300..830nm at 5nm. Normalised so that
# S0 + M1*S1 + M2*S2 is exactly 100 at 560nm for every daylight chromaticity.
CIE_DAYLIGHT_5NM = (
    0.04, 0.02, 0, 3.02, 2.26, 1, 6, 4.5, 2, 17.8, 13.45, 3, 29.6, 22.4, 4, 42.45, 32.2, 6.25,
    55.3, 42, 8.5, 56.3, 41.3, 8.15, 57.3, 40.6, 7.8, 59.55, 41.1, 7.25, 61.8, 41.6, 6.7, 61.65,
    39.8, 6, 61.5, 38, 5.3, 65.15, 40.2, 5.7, 68.8, 42.4, 6.1, 66.1, 40.45, 4.55, 63.4, 38.5, 3,
    64.6, 36.75, 2.1, 65.8, 35, 1.2, 80.3, 39.2, 0.05, 94.8, 43.4, -1.1, 99.8, 44.85, -0.8,
    104.8, 46.3, -0.5, 105.35, 45.1, -0.6, 105.9, 43.9, -0.7, 101.35, 40.5, -0.95, 96.8, 37.1,
    -1.2, 105.35, 36.9, -1.9, 113.9, 36.7, -2.6, 119.75, 36.3, -2.75, 125.6, 35.9, -2.9, 125.55,
    34.25, -2.85, 125.5, 32.6, -2.8, 123.4, 30.25, -2.7, 121.3, 27.9, -2.6, 121.3, 26.1, -2.6,
    121.3, 24.3, -2.6, 117.4, 22.2, -2.2, 113.5, 20.1, -1.8, 113.3, 18.15, -1.65, 113.1, 16.2,
    -1.5, 111.95, 14.7, -1.4, 110.8, 13.2, -1.3, 108.65, 10.9, -1.25, 106.5, 8.6, -1.2, 107.65,
    7.35, -1.1, 108.8, 6.1, -1, 107.05, 5.15, -0.75, 105.3, 4.2, -0.5, 104.85, 3.05, -0.4,
    104.4, 1.9, -0.3, 102.2, 0.95, -0.15, 100, -0, 0, 98, -0.8, 0.1, 96, -1.6, 0.2, 95.55,
    -2.55, 0.35, 95.1, -3.5, 0.5, 92.1, -3.5, 1.3, 89.1, -3.5, 2.1, 89.8, -4.65, 2.65, 90.5,
    -5.8, 3.2, 90.4, -6.5, 3.65, 90.3, -7.2, 4.1, 89.35, -7.9, 4.4, 88.4, -8.6, 4.7, 86.2,
    -9.05, 4.9, 84, -9.5, 5.1, 84.55, -10.2, 5.9, 85.1, -10.9, 6.7, 83.5, -10.8, 7, 81.9, -10.7,
    7.3, 82.25, -11.35, 7.95, 82.6, -12, 8.6, 83.75, -13, 9.2, 84.9, -14, 9.8, 83.1, -13.8, 10,
    81.3, -13.6, 10.2, 76.6, -12.8, 9.25, 71.9, -12, 8.3, 73.1, -12.65, 8.95, 74.3, -13.3, 9.6,
    75.35, -13.1, 9.05, 76.4, -12.9, 8.5, 69.85, -11.75, 7.75, 63.3, -10.6, 7, 67.5, -11.1, 7.3,
    71.7, -11.6, 7.6, 74.35, -11.9, 7.8, 77, -12.2, 8, 71.1, -11.2, 7.35, 65.2, -10.2, 6.7,
    56.45, -9, 5.95, 47.7, -7.8, 5.2, 58.15, -9.5, 6.3, 68.6, -11.2, 7.4, 66.8, -10.8, 7.1, 65,
    -10.4, 6.8, 65.5, -10.5, 6.9, 66, -10.6, 7, 63.5, -10.15, 6.7, 61, -9.7, 6.4, 57.15, -9,
    5.95, 53.3, -8.3, 5.5, 56.1, -8.8, 5.8, 58.9, -9.3, 6.1, 60.4, -9.55, 6.3, 61.9, -9.8, 6.5
)

DAYLIGHT_LAMBDA_MIN = 300.0
DAYLIGHT_STEP = 5.0

# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------


def wavelengths():
    return np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)


def cmf():
    """The colour matching functions as a (471, 3) array on the 1nm grid."""
    table = np.array(CIE_XYZ_1NM, dtype=np.float64).reshape(-1, 3)
    if table.shape[0] != LAMBDA_COUNT:
        raise ValueError(f"CMF table has {table.shape[0]} rows, expected {LAMBDA_COUNT}")
    return table


def daylight_basis():
    """S0/S1/S2 resampled from their native 5nm onto the 1nm grid, as (471, 3)."""
    table = np.array(CIE_DAYLIGHT_5NM, dtype=np.float64).reshape(-1, 3)
    source = DAYLIGHT_LAMBDA_MIN + DAYLIGHT_STEP * np.arange(table.shape[0])
    target = wavelengths()
    # Linear resampling. The basis is smooth and slowly varying, so the interpolation error is
    # orders of magnitude below the CMF quantisation it gets multiplied against.
    return np.stack([np.interp(target, source, table[:, i]) for i in range(3)], axis=1)


def daylight_weights(x, y):
    """The (M1, M2) that select the CIE D illuminant of chromaticity (x, y). CIE 15.

    The denominator runs about 0.0034 *on* the daylight locus, so it is small even where the
    model is valid and the shader has to floor it -- see the sky in shaders/spectral.slangh.
    """
    denominator = 0.0241 + 0.2562 * x - 0.7341 * y
    return ((-1.3515 - 1.7703 * x + 5.9114 * y) / denominator,
            (0.0300 - 31.4424 * x + 30.0717 * y) / denominator)


def daylight_spectrum(basis, x, y):
    """The CIE D illuminant at chromaticity (x, y), reconstructed from the basis.

    This is how CIE *defines* the D series, so D65 is simply this evaluated at D65's
    chromaticity -- no separate D65 table is needed, and the sky in the shader reuses the very
    same basis with a chromaticity Preetham hands it.
    """
    m1, m2 = daylight_weights(x, y)
    return basis[:, 0] + m1 * basis[:, 1] + m2 * basis[:, 2]


def srgb_matrices(white_xyz):
    """Derives sRGB <-> XYZ from the primaries and the given white, rather than hardcoding.

    Deriving means the matrix is exactly consistent with the white point actually used, which
    is what makes an albedo of (1,1,1) resolve back to (1,1,1) rather than to something a
    fraction of a percent off.
    """
    xy = SRGB_PRIMARIES
    z = 1.0 - xy[:, 0] - xy[:, 1]
    primaries = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), z / xy[:, 1]], axis=0)
    scale = np.linalg.solve(primaries, white_xyz)
    rgb_to_xyz = primaries * scale
    return rgb_to_xyz, np.linalg.inv(rgb_to_xyz)


def xyz_to_lab(xyz, white):
    """CIE Lab. The fit minimises here rather than in RGB: Lab spreads the residual
    perceptually, where minimising in RGB over-weights the greens the eye is most sensitive to
    and leaves visible error in the blues."""
    ratio = xyz / white
    delta = 6.0 / 29.0
    cube_root = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 1e-12)),
                         ratio / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack([116.0 * cube_root[..., 1] - 16.0,
                     500.0 * (cube_root[..., 0] - cube_root[..., 1]),
                     200.0 * (cube_root[..., 1] - cube_root[..., 2])], axis=-1)


def xyz_to_lab_jacobian(xyz, white):
    """d(Lab)/d(XYZ), shaped (..., 3, 3). Analytic, because finite differencing it inside a
    Gauss-Newton loop costs three extra spectral integrals per iteration."""
    ratio = xyz / white
    delta = 6.0 / 29.0
    # d(cube_root)/d(ratio), then the chain rule through ratio = xyz / white.
    derivative = np.where(ratio > delta**3,
                          1.0 / (3.0 * np.cbrt(np.maximum(ratio, 1e-12)) ** 2),
                          np.full_like(ratio, 1.0 / (3.0 * delta**2))) / white

    jacobian = np.zeros(xyz.shape[:-1] + (3, 3))
    jacobian[..., 0, 1] = 116.0 * derivative[..., 1]
    jacobian[..., 1, 0] = 500.0 * derivative[..., 0]
    jacobian[..., 1, 1] = -500.0 * derivative[..., 1]
    jacobian[..., 2, 1] = 200.0 * derivative[..., 1]
    jacobian[..., 2, 2] = -200.0 * derivative[..., 2]
    return jacobian


# ---------------------------------------------------------------------------
# the model
#
#   s(t) = c0*t^2 + c1*t + c2,   t = (lambda - 360) / 470
#   S(t) = 1/2 + s / (2*sqrt(1 + s^2))
#
# Deliberate deviation from the published form, which uses lambda in nanometres directly:
# there c0 is around 1e-5 and c0*lambda^2 + c1*lambda + c2 is a sum of large near-cancelling
# terms, which is a catastrophic cancellation trap in fp32 on the GPU and an ill-conditioned
# Gauss-Newton on the CPU. In t all three coefficients are order 1 and both problems go away.
# A table baked here therefore cannot be fed to a renderer expecting the nanometre form.
# ---------------------------------------------------------------------------


def basis_powers():
    """The (3, 471) matrix of (t^2, t, 1), which is both the model's basis and ds/dc."""
    t = (wavelengths() - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
    return np.stack([t * t, t, np.ones_like(t)], axis=0)


def sigmoid(s):
    return 0.5 + s / (2.0 * np.sqrt(1.0 + s * s))


def sigmoid_derivative(s):
    return 0.5 / np.power(1.0 + s * s, 1.5)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def z_nodes(resolution):
    """The non-uniform scale for the largest-component axis.

    Denser at both ends, where the coefficients move fastest: near 0 the sigmoid has to be
    driven far negative to make a dark colour, and near 1 far positive. The exact curve is a
    free choice, which is precisely why it is written into the blob rather than assumed -- the
    shader searches these nodes, so changing it here is a re-bake and nothing more.
    """
    t = np.linspace(0.0, 1.0, resolution)
    return t * t * (3.0 - 2.0 * t)


def grid_targets(slice_index, z, resolution):
    """The (resolution^2, 3) linear-sRGB colours one z level of one slice stands for.

    `slice_index` is which channel is the largest; the other two are stored as ratios to it,
    so the whole [0,1]^3 cube is covered exactly once by the three slices.
    """
    axis = np.linspace(0.0, 1.0, resolution)
    y_ratio, x_ratio = np.meshgrid(axis, axis, indexing="ij")

    rgb = np.zeros((resolution, resolution, 3))
    rgb[..., slice_index] = z
    rgb[..., (slice_index + 1) % 3] = x_ratio * z
    rgb[..., (slice_index + 2) % 3] = y_ratio * z
    return rgb.reshape(-1, 3)


def fit_level(targets_lab, coefficients, operator, powers, white, iterations, damping):
    """Gauss-Newton for one z level: every (x, y) cell of one slice, solved at once.

    `operator` is the (3, 471) map from a reflectance spectrum to XYZ -- the illuminant and
    the colour matching functions already folded together and normalised, so a reflectance of
    1 everywhere integrates to Y = 1.
    """
    for _ in range(iterations):
        s = coefficients @ powers                      # (N, 471)
        reflectance = sigmoid(s)
        xyz = reflectance @ operator.T                 # (N, 3)

        residual = xyz_to_lab(xyz, white) - targets_lab

        # d(XYZ)/d(c) = sum over lambda of operator * dS/ds * ds/dc.
        weighted = sigmoid_derivative(s)               # (N, 471)
        d_xyz = np.einsum("cl,nl,kl->nck", operator, weighted, powers)
        jacobian = xyz_to_lab_jacobian(xyz, white) @ d_xyz          # (N, 3, 3)

        # Levenberg damping, so a cell whose Jacobian is near singular -- which happens where
        # the sigmoid saturates and the spectrum stops responding to its coefficients -- steps
        # short instead of flying off.
        jtj = np.einsum("nki,nkj->nij", jacobian, jacobian)
        jtr = np.einsum("nki,nk->ni", jacobian, residual)
        jtj[:, [0, 1, 2], [0, 1, 2]] += damping

        # The explicit trailing axis is required: numpy 2 reads a stacked solve's (N, 3)
        # right-hand side as one (m, n) matrix rather than as N vectors.
        coefficients = coefficients - np.linalg.solve(jtj, jtr[..., None])[..., 0]
        # The sigmoid saturates well before this; the bound only stops a diverging cell from
        # reaching inf and poisoning its neighbours through the warm start.
        coefficients = np.clip(coefficients, -1000.0, 1000.0)

    return coefficients


def fit_slice(slice_index, nodes, operator, powers, white, rgb_to_xyz, resolution, iterations,
              damping, verbose):
    """One slice, marching outward in z from the middle.

    The march is not an optimisation: cold-starting at z near 0 or 1 diverges, because the
    spectrum there has to be driven almost flat-dark or flat-bright and the initial Jacobian
    carries no useful direction. Starting from the middle, where the fit is easy, and handing
    each level's answer to its neighbour keeps every solve in a good basin.
    """
    cells = resolution * resolution
    coefficients = np.zeros((resolution, cells, 3))
    middle = resolution // 2

    for direction in (1, -1):
        warm = np.zeros((cells, 3))
        indices = range(middle, resolution) if direction > 0 else range(middle - 1, -1, -1)

        for level in indices:
            # A pure black target is unrepresentable -- it needs the sigmoid driven to minus
            # infinity -- so the darkest level is nudged off zero. The error this leaves is on
            # a colour that is black to well past eight bits.
            z = max(nodes[level], 1e-4)
            targets = grid_targets(slice_index, z, resolution)
            targets_lab = xyz_to_lab(targets @ rgb_to_xyz.T, white)

            warm = fit_level(targets_lab, warm.copy(), operator, powers, white, iterations,
                             damping)
            coefficients[level] = warm

        if verbose:
            print(f"  slice {slice_index} {'up' if direction > 0 else 'down'} done")

    return coefficients


def evaluate(coefficients, operator, powers, white, targets_rgb, rgb_to_xyz, xyz_to_rgb):
    """Re-integrates a fitted block and reports how far off it landed, in dE and in RGB.

    This is the gate that matters: it catches a diverged Gauss-Newton at bake time, before a
    single ray has been traced against a table that looks plausible and is wrong.
    """
    reflectance = sigmoid(coefficients @ powers)
    xyz = reflectance @ operator.T
    delta_e = np.sqrt(np.sum((xyz_to_lab(xyz, white)
                              - xyz_to_lab(targets_rgb @ rgb_to_xyz.T, white)) ** 2, axis=-1))
    rgb_error = np.abs(xyz @ xyz_to_rgb.T - targets_rgb)
    return delta_e, rgb_error


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

# "<8sIIIIff3fI48s" -- magic, version, resolution, coefficient count, lambda count,
# lambda min/max, the three daylight luminance constants, flags, name.
HEADER_FORMAT = "<8sIIIIff3fI%ds" % NAME_SIZE
assert struct.calcsize(HEADER_FORMAT) <= HEADER_SIZE


def write_blob(path, resolution, nodes, cmf_table, basis, luminance, xyz_to_rgb, d65_m,
               coefficients, verbose):
    """Header, then every constant the shader needs, then the four tables.

    The small constants lead the payload rather than living in the header, so the renderer can
    upload the whole thing verbatim and reach all of it through one device address -- see the
    layout comment in src/spectral.h, which this order defines.

    Written to a temporary and moved into place, so an interrupted bake can never leave a
    half written blob that passes the magic check -- the same guarantee tools/bake_assets.py
    gives for a model.
    """
    header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, resolution, 3, LAMBDA_COUNT,
                         LAMBDA_MIN, LAMBDA_MAX, *luminance, 0,
                         b"sRGB / D65 / Jakob-Hanika 2019, t basis")

    temporary = path + ".tmp"
    with open(temporary, "wb") as file:
        file.write(header)
        file.write(b"\0" * (HEADER_SIZE - len(header)))
        for block in (xyz_to_rgb.reshape(-1), np.asarray(d65_m), nodes, cmf_table, basis,
                      coefficients):
            file.write(np.ascontiguousarray(block, dtype="<f4").tobytes())

    os.replace(temporary, path)
    if verbose:
        print(f"  wrote {path} ({os.path.getsize(path) / (1 << 20):.2f} MiB)")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def bake(destination, resolution, iterations, damping, tolerance, verbose):
    started = time.time()

    cmf_table = cmf()
    basis = daylight_basis()

    # D65 is the daylight basis at D65's chromaticity -- that is CIE's own definition of it --
    # so the illuminant the LUT is fitted under and the illuminant the sky reconstructs come
    # from one table and cannot drift apart.
    d65 = daylight_spectrum(basis, *D65_CHROMATICITY)

    # Normalised so a perfect white reflector integrates to Y = 1. Every absolute level in the
    # renderer hangs off this one choice.
    white_scale = 1.0 / np.sum(d65 * cmf_table[:, 1])
    operator = (cmf_table * d65[:, None]).T * white_scale        # (3, 471), reflectance -> XYZ
    white = operator @ np.ones(LAMBDA_COUNT)                     # XYZ of that white reflector

    rgb_to_xyz, xyz_to_rgb = srgb_matrices(white)

    if verbose:
        chromaticity = white[:2] / np.sum(white)
        print(f"  white point   XYZ {white[0]:.6f} {white[1]:.6f} {white[2]:.6f}")
        print(f"                xy  {chromaticity[0]:.6f} {chromaticity[1]:.6f}"
              f"  (D65 is {D65_CHROMATICITY[0]} {D65_CHROMATICITY[1]})")
        # Worth eyeballing against the published sRGB matrix: the top row should be close to
        # 3.2406 -1.5372 -0.4986. It will not match to the last digit, because it is derived
        # from the white point this bake actually computed rather than from the standard's
        # rounded one -- which is the point.
        print("  XYZ -> linear sRGB")
        for row in xyz_to_rgb:
            print(f"                {row[0]:9.6f} {row[1]:9.6f} {row[2]:9.6f}")

    # The basis is stored with `white_scale` already folded in, so a daylight spectrum
    # reconstructed from it needs no further normalisation anywhere: reconstructing at D65's
    # chromaticity gives an illuminant under which a white reflector integrates to Y = 1, and
    # the shader's resolve is then a plain sum against the colour matching functions with no
    # trailing constant. Storing the raw basis would force that constant into the shader and
    # into every other consumer, where it could drift out of step with this one.
    basis = basis * white_scale

    # The sky needs the luminance of each basis function separately: Preetham hands the shader
    # a target Y, and Y is linear in (1, M1, M2), so these three constants replace a spectral
    # integral per ray with a dot product. In these units D65 itself has Y exactly 1, which is
    # what makes the sky's rescale and the lights' brightness the same currency.
    luminance = tuple(float(np.sum(basis[:, i] * cmf_table[:, 1])) for i in range(3))

    powers = basis_powers()
    nodes = z_nodes(resolution)

    blocks = []
    worst = 0.0
    total = 0.0
    count = 0
    for slice_index in range(3):
        fitted = fit_slice(slice_index, nodes, operator, powers, white, rgb_to_xyz, resolution,
                           iterations, damping, verbose)

        targets = np.stack([grid_targets(slice_index, max(nodes[level], 1e-4), resolution)
                            for level in range(resolution)])
        delta_e, _ = evaluate(fitted, operator, powers, white, targets, rgb_to_xyz, xyz_to_rgb)

        worst = max(worst, float(np.max(delta_e)))
        total += float(np.sum(delta_e))
        count += delta_e.size
        if verbose:
            print(f"  slice {slice_index}  mean dE {np.mean(delta_e):.4f}"
                  f"  max dE {np.max(delta_e):.4f}")
        blocks.append(fitted)

    print(f"  fit {count} cells in {time.time() - started:.1f}s"
          f"  mean dE {total / count:.4f}  max dE {worst:.4f}")

    if worst > tolerance:
        # Refusing to write is the point: a diverged fit produces a table that loads cleanly,
        # renders, and is quietly wrong. Far better to fail here than to debug it from an image.
        print(f"bake_spectral: max dE {worst:.4f} exceeds the tolerance of {tolerance:.4f};"
              f" refusing to write {destination}", file=sys.stderr)
        return False

    write_blob(destination, resolution, nodes, cmf_table, basis, luminance, xyz_to_rgb,
               daylight_weights(*D65_CHROMATICITY), np.stack(blocks), verbose)
    return True


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Bake the renderer's spectral tables.")
    parser.add_argument("--dst", default=os.path.join(root, "assets", "bin"))
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--tolerance", type=float, default=6.0,
                        help="refuse to write if any cell's dE exceeds this")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    destination = os.path.join(args.dst, "spectral.bin")

    # The blob depends only on this script, so its own mtime is the whole dependency set --
    # which keeps a warm build silent, exactly as the model baker does per model.
    if not args.force and os.path.isfile(destination):
        if os.path.getmtime(destination) >= os.path.getmtime(os.path.abspath(__file__)):
            if args.verbose:
                print(f"  {destination} is up to date")
            return 0

    print("baking spectral tables")
    return 0 if bake(destination, args.resolution, args.iterations, args.damping,
                     args.tolerance, args.verbose) else 1


if __name__ == "__main__":
    sys.exit(main())
