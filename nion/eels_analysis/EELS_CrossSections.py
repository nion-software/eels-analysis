"""
    EELS Cross Section

    A library of functions for computing EELS edge cross-sections.
"""

import numpy


def k_shell_hydrogenic_gos(atomic_number: int, edge_onset_eV: float, edge_delta_eV: float,
                            beam_energy_eV: float, collection_angle_rad: float) -> numpy.ndarray:
    """Return the K-shell generalized oscillator strength (GOS) calculated on the hydrogenic model as an ndarray.

    This algorithm is based on the hydrogenic model formulation given by Egerton in chapter 3 of his book
    entitled Electron Energy-Loss Spectroscopy in the Electron Microscope (in its 3rd edition as of 2011).
    Egerton's original expressions (formula numbers given below) have been slightly reformulated by Mike Kundmann
    to minimize the fundamental constants required, eliminate unnecessary approximations in the relativistic
    kinematics, and simplify the GOS expressions.  In order for the angular portion of any subsequent cross-section
    computation to be carried out via straightforward NumPy array integration with fixed limits of integration,
    the GOS table is computed versus scattering angle, theta, rather than (dimensionless) momentum transfer, qa0.

    The returned GOS array has the following properties:
      0-axis is the energy loss dimension, from edge_onset_eV through edge_onset_eV + edge_delta_eV
      1-axis is the scattering angle dimension, from 0 through collection_angle_rad
      intensity is in units of 1 / (eV * steradian)
    """
    electronRestEnergy_eV = 510999.0
    fineStructureConstant = 1 / 137.036
    rydbergEnergy_eV = 0.5 * electronRestEnergy_eV * fineStructureConstant ** 2

    beamGamma = 1 + beam_energy_eV / electronRestEnergy_eV
    beamBeta2 = 1 - 1 / beamGamma ** 2

    energySampleCount = numpy.fmin(numpy.fmax(50, 100 * numpy.round(edge_delta_eV / 100)), 1000) + 1
    thetaSampleCount = int(numpy.fmin(numpy.fmax(50, 100 * numpy.round(collection_angle_rad / 0.050)), 400) + 1)

    screenedZ2 = 1.
    shellOccupancy = 1
    if atomic_number > 1:
        screenedZ2 = (atomic_number - 0.5) ** 2
        shellOccupancy = 2

    # Generate epsilon array (scaled energy-loss) = E/m0c^2 = E/Me over requested energy loss range
    epsilon = numpy.linspace(edge_onset_eV, edge_onset_eV + edge_delta_eV, energySampleCount, dtype = numpy.float64) / electronRestEnergy_eV

    # Generate corresponding phiE array = 1-k1/k0 ~= thetaE
    phiE = 1 - numpy.sqrt(1 - 2 * epsilon * (beamGamma - epsilon / 2) / (beamGamma ** 2 * beamBeta2))

    # Generate thetaTerm array = 4sin(theta/2)^2 for requested collection angle
    thetaTerm = numpy.linspace(0, collection_angle_rad, thetaSampleCount, dtype = numpy.float64)
    thetaTerm = 4 * numpy.sin(thetaTerm / 2) ** 2

    # Generate Q^2 map = K0^2[phiE^2 + 4(1-phiE)sin(theta/2)^2], where
    # Q = qa0, K0 = k0a0 = gamma0*beta0/alpha, and alpha = fine structure constant.
    # This is an exact reformulation of Egerton's equation 3.141, making approximations 3.144 and 3.146 unnecessary.
    # A complete 2D GOS map is generated thanks to Python broadcasting and judicious shaping of the thetaTerm array.
    Q2 = (phiE ** 2 + (1 - phiE) * thetaTerm.reshape(thetaSampleCount, 1)) * beamBeta2 * (beamGamma / fineStructureConstant) ** 2

    # Generate epsilonR array (energy-loss in Rydbergs) = E/Ry = 2*epsilon/alpha^2
    epsilonR = 2 * epsilon / fineStructureConstant ** 2

    # Generate common GOS pre-factor map = 128*Ne(E/Ry)(Q^2+(E/Ry)/3)/[((Q^2-E/Ry)/Zs)^2+4*Q^2]^3/Ry
    # This is an exact reformulation of the common factor in Egerton's equations 3.125 and 3.126.
    # His equations assume 2 electrons (Ne = 2) occupy the 1s shell, which is not true for hydrogen.
    # We compensate with the shellOccupancy factor, following Egerton's use of RNK in his SIMGAK3 program.
    gos = 128 * shellOccupancy * epsilonR * (Q2 + epsilonR / 3) / ((Q2 - epsilonR) ** 2 / screenedZ2 + 4 * Q2) ** 3 / rydbergEnergy_eV

    # Generate kH array = (|(E/Ry)/Zs^2) - 1|)^(1/2)
    # To simplify Egerton equation 3.127, we take kH to be the absolute value of that defined in equation 3.124.
    # We avoid computational overflow in the GOS exponential factors by limiting the smallest value of kH to no less than 0.01.
    # This yields the correct kH -> 0 limiting value of the exponential factor in 3.125 and 3.126: exp(-4*Zs^2/(Q^2-E/Ry+2Zs^2))
    kH = numpy.fmax(numpy.sqrt(numpy.fabs(epsilonR / screenedZ2 - 1)), 0.01)

    # Determine energy array index at which "free" states begin
    boundStateMask = numpy.zeros_like(epsilonR)
    boundStateMask[epsilonR < screenedZ2] = 1
    freeStateStartIndex = int(boundStateMask.sum())

    # Generate GOS 'exponential' factor map, i.e. the exponential factors in Egerton's equations 3.125 and 3.126
    gosFactor = numpy.zeros_like(Q2)

    # Compute bound-state portion = exp(-y), as in 3.126.
    # Note that -y has been reformulated as log[(Q^2-E/Ry+2(1-kH)Zs^2)/(Q^2-E/Ry+2(1+kH)Zs^2))/kH.
    epsilonR_bound = epsilonR[:freeStateStartIndex]
    kH_bound = kH[:freeStateStartIndex]
    Q2_bound = Q2[:, :freeStateStartIndex]
    gosFactor_bound = gosFactor[:, :freeStateStartIndex]
    gosFactor_bound[...] = Q2_bound - epsilonR_bound + 2 * screenedZ2 * (1 - kH_bound)
    gosFactor_bound[...] /= Q2_bound - epsilonR_bound + 2 * screenedZ2 * (1 + kH_bound)
    gosFactor_bound[...] = numpy.exp(numpy.log(gosFactor_bound) / kH_bound)

    if freeStateStartIndex < energySampleCount:
        # Compute free-state portion = exp(-2*betaPrime/kH)/[1-exp(-2*pi/kH)], as in 3.125.
        # Note that betaPrime has been reformulated as arctan(2*Zs^2kH/(Q^2 - E/Ry + 2*Zs^2).
        epsilonR_free = epsilonR[freeStateStartIndex:]
        kH_free = kH[freeStateStartIndex:]
        Q2_free = Q2[:, freeStateStartIndex:]
        gosFactor_free = gosFactor[:, freeStateStartIndex:]
        gosFactor_free[...] = numpy.arctan2(2 * screenedZ2 * kH_free, Q2_free - epsilonR_free + 2 * screenedZ2)
        gosFactor_free[...] = numpy.exp(-2 * gosFactor_free / kH_free) / (1 - numpy.exp(-2 * numpy.pi / kH_free))

    gos *= gosFactor

    return gos


def generalized_oscillator_strength(atomic_number: int, shell_number: int, subshell_index: int, edge_onset_eV: float, edge_delta_eV: float,
                                        beam_energy_eV: float, collection_angle_rad: float) -> numpy.ndarray:
    """Return the generalized oscillator strength (GOS) for the specified electron shell as an ndarray.

    In order for the angular portion of any subsequent cross-section computation to be carried out via straightforward
    NumPy array integration with fixed limits of integration, the GOS table is computed versus scattering angle, theta,
    rather than (dimensionless) momentum transfer, qa0.

    The returned GOS array has the following properties:
      0-axis is the energy loss dimension, from edge_onset_eV through edge_onset_eV + edge_delta_eV
      1-axis is the scattering angle dimension, from 0 through collection_angle_rad
      intensity is in units of 1 / (eV * steradian)
    """
    assert atomic_number >= 1
    assert shell_number == 1
    assert subshell_index == 1

    if shell_number == 1:
        gos = k_shell_hydrogenic_gos(atomic_number, edge_onset_eV, edge_delta_eV, beam_energy_eV, collection_angle_rad)
    else:
        print("No GOS routine available for L, M, N, and O electron shells.")

    return gos


def kohl_collection_efficiency(theta_rad: numpy.ndarray, alpha_rad: float, beta_rad: float) -> numpy.ndarray:
    """Return the Kohl collection efficiency for events with scattering angles (in radians) given by the theta array.

    alpha_rad is the convergence semi-angle of the incident beam in radians.
    beta_rad is the collection semi-angle (usually defined by the spectrometer entrance aperture) in radians.

    This algorithm is based on the aperture cross-correlation method described by H. Kohl in Ultramicroscopy 16 (1985) 265-268.
    In effect, it computes the fraction of scattering events at angle theta that manage to enter the collection aperture in terms
    of the overlapping angular area of the convergence and collection apertures. In this implementation, the overlap area is
    divided by the smaller of the convergence and collection aperture angular areas, thereby yielding the collection efficiency
    with respect to the captured total spectrum intensity. This is a departure from Kohl's published expression, which gives the
    collection efficiency with respect to the incident beam intensity (even when alpha > beta and much of the incident beam is blocked).
    """
    assert theta_rad.ndim == 1

    sum_angle = alpha_rad + beta_rad
    diff_angle = numpy.fabs(alpha_rad - beta_rad)

    # Bypass the (negligible) overlap area correction if either aperture has an angular radius 1% or less of the other.
    # In this case, the expressions below are not well-conditioned against round-off error anyway, so best not to apply them.
    if alpha_rad <= beta_rad / 100:
        sum_angle = beta_rad
        diff_angle = beta_rad
    elif beta_rad <= alpha_rad / 100:
        sum_angle = alpha_rad
        diff_angle = alpha_rad

    # Determine theta range requiring aperture overlap computation, i.e. that with non-zero efficiency < 1.
    theta_start = theta_rad[theta_rad <= diff_angle].size
    theta_end = theta_rad[theta_rad < sum_angle].size

    # Determine the collection efficiency array.
    collection_efficiency = numpy.zeros_like(theta_rad)
    collection_efficiency[:theta_start] = 1
    if theta_end > theta_start:
        theta_slice = theta_rad[theta_start:theta_end]
        efficiency_slice = collection_efficiency[theta_start:theta_end]

        # Start with the sum of the intersecting sector areas
        efficiency_slice[...] = numpy.arccos((1 + beta_rad / theta_slice) * (theta_slice - beta_rad)/(2 * alpha_rad) + alpha_rad / (2 * theta_slice)) * alpha_rad ** 2
        efficiency_slice[...] += numpy.arccos((1 + alpha_rad / theta_slice) * (theta_slice - alpha_rad)/(2 * beta_rad) + beta_rad / (2 * theta_slice)) * beta_rad ** 2

        # Subtract the "directional" sector triangles to yield the area of the overlapping "lens" region of the two angular apertures
        efficiency_slice[...] -= numpy.sqrt((sum_angle - theta_slice) * (sum_angle + theta_slice) * (theta_slice - diff_angle) * (theta_slice + diff_angle)) / 2

        # Divide by the smallest aperture angular area to yield a spectrum collection efficiency in the range (0, 1)
        efficiency_slice[...] /= numpy.pi * min(alpha_rad, beta_rad) ** 2

    return collection_efficiency


def energy_diff_cross_section_nm2_per_eV(atomic_number: int, shell_number: int, subshell_index: int,
                                         edge_onset_eV: float, edge_delta_eV: float, beam_energy_eV: float,
                                         convergence_angle_rad: float, collection_angle_rad: float) -> numpy.ndarray:
    """Return the energy differential cross section for the specified electron shell and experimental parameters.

    Uses generalized_oscillator_strength and kohl_collection_efficiency functions.

    This algorithm is based on the Bethe theory formulation given by Egerton in chapter 3 of his book entitled
    Electron Energy-Loss Spectroscopy in the Electron Microscope (in its 3rd edition as of 2011).
    Egerton's original expressions (formula numbers given below) have been slightly reformulated by Mike Kundmann
    to minimize the number of fundamental constants, eliminate unnecessary approximations in the relativistic
    kinematics, and to simplify the angular integration of the GOS via efficient NumPy array-based techniques.
    In particular, note that the latter is carried out with respect to scattering angle theta, rather than log(qa0^2).
    This also simplifies the convergence angle correction, which follows Kohl's aperture cross-correlation approach.

    The returned differential cross-section value is in units of nm * nm / eV.
    """
    assert beam_energy_eV > 0
    assert edge_onset_eV > 0
    assert edge_delta_eV > 0
    assert convergence_angle_rad >= 0
    assert collection_angle_rad > 0

    hBarC_eV_nm = 197.325
    electronRestEnergy_eV = 510999.0
    fineStructureConstant = 1 / 137.036
    bohrRadius_nm = hBarC_eV_nm / (fineStructureConstant * electronRestEnergy_eV)

    beamGamma = 1 + beam_energy_eV / electronRestEnergy_eV
    beamBeta2 = 1 - 1 / beamGamma ** 2

    max_scattering_angle_rad = convergence_angle_rad + collection_angle_rad
    gos = generalized_oscillator_strength(atomic_number, shell_number, subshell_index,
                                          edge_onset_eV, edge_delta_eV, beam_energy_eV, max_scattering_angle_rad)

    energySampleCount = gos.shape[1]
    thetaSampleCount = gos.shape[0]

    # Generate epsilon array (scaled energy-loss) = E/m0c^2 = E/Me over requested energy loss range
    epsilon = numpy.linspace(edge_onset_eV, edge_onset_eV + edge_delta_eV, energySampleCount, dtype = numpy.float64) / electronRestEnergy_eV

    # Generate corresponding phiE array = 1-k1/k0 ~= thetaE
    phiE = 1 - numpy.sqrt(1 - 2 * epsilon * (beamGamma - epsilon / 2) / (beamGamma ** 2 * beamBeta2))

    # Generate appropriate theta array for maximum scattering angle
    theta_rad = numpy.linspace(0, max_scattering_angle_rad, thetaSampleCount, dtype = numpy.float64)

    # Generate Q^2 map = K0^2[phiE^2 + 4(1-phiE)sin(theta/2)^2], where
    # Q = qa0, K0 = k0a0 = gamma0*beta0/alpha, and alpha = fine structure constant.
    # This is an exact reformulation of Egerton's equation 3.141, making approximations 3.144 and 3.146 unnecessary.
    # A complete 2D GOS map is generated thanks to Python broadcasting and judicious shaping of the thetaTerm array.
    thetaTerm = 4 * numpy.sin(theta_rad.reshape(thetaSampleCount, 1) / 2) ** 2
    Q2 = (phiE ** 2 + (1 - phiE) * thetaTerm) * beamBeta2 * (beamGamma / fineStructureConstant) ** 2

    # Generate differential cross-section map = 2(1-phiE)(alpha*gamma0*a0)^2/((E/Me)Q^2) * df/dE, where a0 = Bohr radius.
    # This is a slightly reformulated, but exactly equivalent, version of Egerton's equation 3.26.
    dSigma = 2 * (1 - phiE) * (fineStructureConstant * beamGamma * bohrRadius_nm) ** 2 / (epsilon * Q2) * gos

    # Integrate over solid angle out to collection angle to yield dSigma/dE,
    # Apply collection efficiency factor to correct for convergence angle via the Kohl method.
    collection_efficiency = kohl_collection_efficiency(theta_rad, convergence_angle_rad, collection_angle_rad)
    dSigma *= 2 * numpy.pi * theta_rad.reshape(thetaSampleCount, 1) * collection_efficiency.reshape(thetaSampleCount, 1)
    theta_step = max_scattering_angle_rad / (thetaSampleCount - 1)
    energyDiffSigma = numpy.trapz(dSigma, dx = theta_step, axis = 0)

    return energyDiffSigma


def partial_cross_section_nm2(atomic_number: int, shell_number: int, subshell_index: int,
                              edge_onset_eV: float, edge_delta_eV: float, beam_energy_eV: float,
                              convergence_angle_rad: float, collection_angle_rad: float) -> float:
    """Return the partial cross section for the specified electron shell and experimental parameters.

    Uses energy_diff_cross_section_nm2_per_eV function.

    The returned cross-section value is in units of nm * nm.
    """

    # Generate the energy differential cross-section array.
    energyDiffSigma = energy_diff_cross_section_nm2_per_eV(atomic_number, shell_number, subshell_index,
                                                           edge_onset_eV, edge_delta_eV, beam_energy_eV,
                                                           convergence_angle_rad, collection_angle_rad)

    # Integrate over energy window to get partial cross-section
    energySampleCount = energyDiffSigma.shape[0]
    energy_step = edge_delta_eV / (energySampleCount - 1)
    partialCrossSection = numpy.trapz(energyDiffSigma, dx = energy_step)

    return partialCrossSection
