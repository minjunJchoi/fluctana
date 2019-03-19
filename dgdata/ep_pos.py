#
def get_ep_pos():
    ep_rpos = {}
    ep_zpos = {}
    with open('./ep_pos.dat', 'r') as flines:
        for f in flines:
            if f.strip().split()[0] == 'R(m)':
                continue

            R = float(f.strip().split()[0])
            z = float(f.strip().split()[1])
            i = float(f.strip().split()[2])

            chname = 'EP{:02g}'.format(i)

            ep_rpos[chname] = R
            ep_zpos[chname] = z

    return ep_rpos, ep_zpos
