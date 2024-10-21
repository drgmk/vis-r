# download the AS 209 continnum data from here
# https://almascience.eso.org/almadata/lp/DSHARP/
#
# Then split the DSHARP component out and average to 8 channels per SPW
# (inconvenient to include the older data as they have different numbers of SPWs)
split(vis='AS209_continuum.ms/', outputvis='AS209_continuum.split.8ch.ms',
      spw='13,14,15,16,17,18,19,20,21,22,23,24',
      width=[1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1],
      datacolumn='DATA', keepflags=False)

# then export these to be read in by vis-r,
# the exported file is nearly 1GB so not included here
