// Solution from:
//http://lethalman.blogspot.com/2011/08/probability-of-union-of-independent.html
//https://math.stackexchange.com/questions/2341408/proving-an-equality-regarding-fully-independent-events
double inclusion_exclusion(double *a, size_t len) {
    double x = 0;
    for (size_t i = 0; i < len; ++i) {
        x += a[i] * (1 - x);
    }
    return x;
}

