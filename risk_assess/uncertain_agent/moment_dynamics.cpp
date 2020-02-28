#include <iostream>

struct ExogenousMoments{
    double E_wv;
    double E2_wv;
    double E_c;
    double E2_c;
    double E_s;
    double E2_s;
    double E_cs;
    double E_cw;
    double E_sw;
};

struct AMS{
    double E_v;
    double E2_v;
    double E_xs;
    double E_ys;
    double E_xc;
    double E_yc;
    double E_xvs;
    double E_xvc;
    double E_yvs;
    double E_yvc;
    double E_x;
    double E_y;
    double E_xy;
    double E2_x;
    double E2_y;
};



extern "C"{
    double add(const double x, const double y){
        return x + y;
    }

    AMS propagate_moments(const AMS previous, const ExogenousMoments exog_moments){

        // Moments from the previous state.
        const double E_v = previous.E_v;
        const double E2_v = previous.E2_v;
        const double E_xs = previous.E_xs;
        const double E_ys = previous.E_ys;
        const double E_xc = previous.E_xc;
        const double E_yc = previous.E_yc;
        const double E_xvs = previous.E_xvs;
        const double E_xvc = previous.E_xvc;
        const double E_yvs = previous.E_yvs;
        const double E_yvc = previous.E_yvc;
        const double E_x = previous.E_x;
        const double E_y = previous.E_y;
        const double E_xy = previous.E_xy;
        const double E2_x = previous.E2_x;
        const double E2_y = previous.E2_y;

        // Moments from new sources of randomness.
        const double E_wv= exog_moments.E_wv;
        const double E2_wv= exog_moments.E2_wv;
        const double E_c= exog_moments.E_c;
        const double E2_c= exog_moments.E2_c;
        const double E_s= exog_moments.E_s;
        const double E2_s= exog_moments.E2_s;
        const double E_cs= exog_moments.E_cs;
        const double E_cw= exog_moments.E_cw;
        const double E_sw= exog_moments.E_sw;

        AMS new_state;

        new_state.E_v = E_v + E_wv;

        new_state.E2_v = E2_v + 2 * E_v * E_wv + E2_wv;

        new_state.E_xs = E_v*E_cs*E_cw + E_v*E_sw*E2_c + E_xs*E_cw + E_xc * E_sw;

        new_state.E_ys = E_v*E2_s*E_cw + E_v*E_sw*E_cs + E_ys*E_cw + E_yc * E_sw;

        new_state.E_xc = -E_v*E_cs * E_sw + E_v*E2_c*E_cw - E_xs*E_sw + E_xc*E_cw;

        new_state.E_yc = -E_v*E2_s*E_sw + E_v*E_cs*E_cw - E_ys*E_sw + E_yc*E_cw;

        new_state.E_xvs = E2_v*E_cs*E_cw + E2_v*E_sw*E2_c+ E_v*E_wv*E_cs*E_cw + E_v*E_wv*E_sw*E2_c + E_xvs*E_cw +
            E_xvc * E_sw + E_wv * E_xs * E_cw  + E_wv * E_xc * E_sw;

        new_state.E_xvc = -E2_v*E_cs*E_sw + E2_v*E2_c*E_cw- E_v*E_wv*E_cs*E_sw+ E_v*E_wv*E2_c*E_cw -
            E_xvs*E_sw + E_xvc*E_cw- E_wv*E_xs*E_sw + E_wv*E_xc*E_cw;

        new_state.E_yvs = E2_v*E2_s*E_cw+ E2_v*E_cs*E_sw + E_v*E_wv*E2_s*E_cw+ E_v*E_wv*E_cs*E_sw +
            E_yvs*E_cw+ E_yvc*E_sw + E_wv*E_ys*E_cw+ E_wv*E_yc*E_sw;

        new_state.E_yvc = -E2_v*E2_s*E_sw + E2_v*E_cs*E_cw- E_v*E_wv*E2_s*E_sw + E_v*E_wv*E_cs*E_cw-
            E_yvs*E_sw + E_yvc*E_cw- E_wv*E_ys*E_sw + E_wv*E_yc*E_cw;
        
        new_state.E_x = E_x + E_v * E_c;

        new_state.E_y = E_y + E_v * E_s;

        new_state.E_xy = E2_v * E_cs + E_xvs + E_yvc + E_xy;

        new_state.E2_x = E2_v*E2_c + 2*E_xvc + E2_x;

        new_state.E2_y = E2_v*E2_s + 2*E_yvs + E2_y;
        return new_state;
    }
}