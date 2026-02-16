use std::ops;
use std::fmt;

pub fn lerp(start: f32, end: f32, tmin: f32, tmax: f32, t: f32) -> f32 {
    start*(t-tmax)/(tmin-tmax) + end*(tmin-t)/(tmin-tmax)
}

#[derive(Debug, Clone, Copy)]
pub struct VecN<const N: usize> {
    pub elems: [f32; N],
}

pub type Vec2 = VecN<2>;
pub type Vec3 = VecN<3>;
pub type Vec4 = VecN<4>;

macro_rules! vec2 {
    [$x:expr, $y:expr] => {
        crate::math::Vec2{elems: [$x, $y]}
    }
}

macro_rules! vec3 {
    [$x:expr, $y:expr, $z:expr] => {
        crate::math::Vec3{elems: [$x, $y, $z]}
    }
}

macro_rules! vec4 {
    [$x:expr, $y:expr, $z:expr, $w:expr] => {
        crate::math::Vec4{elems: [$x, $y, $z, $w]}
    }
}

pub(crate) use vec2;
pub(crate) use vec3;
pub(crate) use vec4;

impl<const N: usize> VecN<N> {
    pub fn zero() -> Self {
        VecN::<N>::splat(0.0)
    }

    pub fn splat(x: f32) -> Self {
        Self{elems: [x; N]}
    }

    pub fn binop<Binop: Fn(f32, f32) -> f32>(u: Self, v: Self, op: Binop) -> Self {
        (0..N).map(|i| op(u[i], v[i])).collect()
    }

    pub fn unop<Unop: Fn(f32) -> f32>(u: Self, op: Unop) -> Self {
        (0..N).map(|i| op(u[i])).collect()
    }

    // pub fn mul(u: Self, v:Self) -> Self {
    //     VecN::<N>::binop(u, v, |x, y| x*y)
    // }

    pub fn dot(u: Self, v: Self) -> f32 {
        (0..N).map(|i| u[i]*v[i]).sum()
    }

    pub fn normsq(u: Self) -> f32 {
        (0..N).map(|i| u[i]*u[i]).sum()
    }

    pub fn norm(u: Self) -> f32 {
        f32::sqrt(VecN::<N>::normsq(u))
    }

    pub fn normalize(u: Self) -> Self {
        let invnorm = 1.0 / VecN::<N>::norm(u);
        VecN::<N>::unop(u, |x| x*invnorm)
    }

    pub fn clamp(u: Self, lower: f32, upper: f32) -> Self {
        VecN::<N>::unop(u, |x| f32::clamp(x, lower, upper))
    }

    pub fn distsq(u: VecN<N>, v: VecN<N>) -> f32 {
        VecN::<N>::normsq(u - v)
    }

    pub fn dist(u: VecN<N>, v: VecN<N>) -> f32 {
        VecN::<N>::norm(u - v)
    }

    pub fn lerp(start: VecN<N>, end: VecN<N>, tmin: f32, tmax: f32, t: f32) -> VecN<N> {
        VecN::<N>::binop(start, end, |startx, endx| lerp(startx, endx, tmin, tmax, t))
    }

    pub fn min(u: VecN<N>, v: VecN<N>) -> VecN<N> {
        VecN::<N>::binop(u, v, |x, y| f32::min(x, y))
    }

    pub fn max(u: VecN<N>, v: VecN<N>) -> VecN<N> {
        VecN::<N>::binop(u, v, |x, y| f32::max(x, y))
    }
}

impl<const N: usize> FromIterator<f32> for VecN<N> {
    fn from_iter<I: IntoIterator<Item=f32>>(iter: I) -> Self {
        let mut vec = VecN::<N>::zero();
        let mut index = 0usize;
        for x in iter {
            vec[index] = x;
            index += 1;
        }
        vec
    }
}

impl VecN<3> {
    pub fn cross(u: Self, v: Self) -> Self {
        vec3![
            u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0]
        ]
    }

    pub fn transform_point(u: Self, a: Mat4) -> Self {
        vec3![
            a[0][0]*u[0] + a[0][1]*u[1] + a[0][2]*u[2] + a[0][3],
            a[1][0]*u[0] + a[1][1]*u[1] + a[1][2]*u[2] + a[1][3],
            a[2][0]*u[0] + a[2][1]*u[1] + a[2][2]*u[2] + a[2][3]
        ]
    }

    pub fn transform_dir(u: Self, a: Mat4) -> Self {
        vec3![
            a[0][0]*u[0] + a[0][1]*u[1] + a[0][2]*u[2],
            a[1][0]*u[0] + a[1][1]*u[1] + a[1][2]*u[2],
            a[2][0]*u[0] + a[2][1]*u[1] + a[2][2]*u[2]
        ]
    }

    pub fn to_vec2(&self) -> Vec2 {
        vec2![self[0], self[1]]
    }

    pub fn to_vec4(&self, w: f32) -> Vec4 {
        vec4![self[0], self[1], self[2], w]
    }

    pub fn to_point(&self) -> Vec4 {
        self.to_vec4(1.0)
    }

    pub fn to_dir(&self) -> Vec4 {
        self.to_vec4(0.0)
    }
}

impl VecN<4> {
    pub fn to_vec2(&self) -> Vec2 {
        vec2![self[0], self[1]]
    }

    pub fn to_vec3(&self) -> Vec3 {
        vec3![self[0], self[1], self[2]]
    }
}

impl<const N: usize> fmt::Display for VecN<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> fmt::Result {
        let mut ret: fmt::Result = Ok(());
        for i in 0..N {
            ret = write!(f, "{}", self[i]);
            if i < N-1 {
                ret = write!(f, " ");
            }
        }
        ret
    }
}

impl<const N: usize> ops::Index<usize> for VecN<N> {
    type Output = f32;
    fn index(&self, i: usize) -> &Self::Output {
        return &self.elems[i];
    }
}

impl<const N: usize> ops::IndexMut<usize> for VecN<N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        return &mut self.elems[i];
    }
}

impl <const N: usize> ops::Neg for VecN<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        VecN::<N>::unop(self, |x| -x)
    }
}

impl<const N: usize> ops::Add for VecN<N> {
    type Output = Self;
    fn add(self, v: Self) -> Self {
        VecN::<N>::binop(self, v, |x, y| x + y)
    }
}

impl<const N: usize> ops::Sub for VecN<N> {
    type Output = Self;
    fn sub(self, v: Self) -> Self {
        VecN::<N>::binop(self, v, |x, y| x - y)
    }
}

impl<const N: usize> ops::Mul<VecN<N>> for VecN<N> {
    type Output = Self;
    fn mul(self, v: VecN<N>) -> Self::Output {
        VecN::<N>::binop(self, v, |x, y| x * y)
    }
}

impl<const N: usize> ops::Mul<f32> for VecN<N> {
    type Output = Self;
    fn mul(self, a: f32) -> Self {
        VecN::<N>::unop(self, |x| x * a)
    }
}

impl<const N: usize> ops::Mul<VecN<N>> for f32  {
    type Output = VecN<N>;
    fn mul(self, v: VecN<N>) -> Self::Output {
        v * self
    }
}

impl<const N: usize> ops::Div<f32> for VecN<N> {
    type Output = Self;
    fn div(self, a: f32) -> Self {
        VecN::<N>::unop(self, |x| x / a)
    }
}

impl<const N: usize> ops::AddAssign for VecN<N> {
    fn add_assign(&mut self, v: Self) {
        for i in 0..N {
            self[i] += v[i];
        }
    }
}

impl<const N: usize> ops::SubAssign for VecN<N> {
    fn sub_assign(&mut self, v: Self) {
        for i in 0..N {
            self[i] -= v[i];
        }
    }
}

impl<const N: usize> ops::MulAssign<f32> for VecN<N> {
    fn mul_assign(&mut self, a: f32) {
        for i in 0..N {
            self[i] *= a;
        }
    }
}


impl<const N: usize> ops::DivAssign<f32> for VecN<N> {
    fn div_assign(&mut self, a: f32) {
        for i in 0..N {
            self[i] /= a;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    pub elems: [f32; 16]
}

macro_rules! mat4 {
    [$a00:expr, $a01:expr, $a02:expr, $a03:expr,
     $a10:expr, $a11:expr, $a12:expr, $a13:expr,
     $a20:expr, $a21:expr, $a22:expr, $a23:expr,
     $a30:expr, $a31:expr, $a32:expr, $a33:expr] => {
        crate::math::Mat4{
            elems: [
                $a00, $a01, $a02, $a03,
                $a10, $a11, $a12, $a13,
                $a20, $a21, $a22, $a23,
                $a30, $a31, $a32, $a33,
            ]
        }
    }
}

pub(crate) use mat4;

impl Mat4 {
    pub fn zero() -> Self {
        crate::math::Mat4{elems: [0.0; 16]}
    }

    pub fn binop<Binop: Fn(f32, f32) -> f32>(a: Self, b: Self, op: Binop) -> Self {
        (0..16).map(|i| op(a.elems[i], b.elems[i])).collect()
    }

    pub fn unop<Unop: Fn(f32) -> f32>(a: Self, op: Unop) -> Self {
        (0..16).map(|i| op(a.elems[i])).collect()
    }

    pub fn print(m: Self, title: &str) {
        println!("{}:", title);
        for i in 0..4 {
            for j in 0..4 {
                print!("{}", m[i][j]);
                if j < 3 {
                    print!(" ");
                }
            }
            println!();
        }
        println!();
    }

    pub fn dist(a: Mat4, b: Mat4) -> f32 {
        let diff = a - b;
        f32::sqrt((0..16).map(|i| diff.elems[i]*diff.elems[i]).sum())
    }

    pub fn transpose(m: Self) -> Self {
        let mut ret = m;
        for row in 1..4 {
            for col in 0..row {
                let tmp = ret[row][col];
                ret[row][col] = ret[col][row];
                ret[col][row] = tmp;
            }
        }
        ret
    }

    pub fn inverse(m: Self) -> Self {
        let (a00, a01, a02, a03) = (m[0][0], m[1][0], m[2][0], m[3][0]);
        let (a10, a11, a12, a13) = (m[0][1], m[1][1], m[2][1], m[3][1]);
        let (a20, a21, a22, a23) = (m[0][2], m[1][2], m[2][2], m[3][2]);
        let (a30, a31, a32, a33) = (m[0][3], m[1][3], m[2][3], m[3][3]);

        let b00 = a00*a11 - a01*a10;
        let b01 = a00*a12 - a02*a10;
        let b02 = a00*a13 - a03*a10;
        let b03 = a01*a12 - a02*a11;
        let b04 = a01*a13 - a03*a11;
        let b05 = a02*a13 - a03*a12;
        let b06 = a20*a31 - a21*a30;
        let b07 = a20*a32 - a22*a30;
        let b08 = a20*a33 - a23*a30;
        let b09 = a21*a32 - a22*a31;
        let b10 = a21*a33 - a23*a31;
        let b11 = a22*a33 - a23*a32;

        let inv_det = 1.0/(b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06);

        let mut result = Mat4::zero();
        result[0][0] = (a11*b11 - a12*b10 + a13*b09)*inv_det;
        result[1][0] = (-a01*b11 + a02*b10 - a03*b09)*inv_det;
        result[2][0] = (a31*b05 - a32*b04 + a33*b03)*inv_det;
        result[3][0] = (-a21*b05 + a22*b04 - a23*b03)*inv_det;
        result[0][1] = (-a10*b11 + a12*b08 - a13*b07)*inv_det;
        result[1][1] = (a00*b11 - a02*b08 + a03*b07)*inv_det;
        result[2][1] = (-a30*b05 + a32*b02 - a33*b01)*inv_det;
        result[3][1] = (a20*b05 - a22*b02 + a23*b01)*inv_det;
        result[0][2] = (a10*b10 - a11*b08 + a13*b06)*inv_det;
        result[1][2] = (-a00*b10 + a01*b08 - a03*b06)*inv_det;
        result[2][2] = (a30*b04 - a31*b02 + a33*b00)*inv_det;
        result[3][2] = (-a20*b04 + a21*b02 - a23*b00)*inv_det;
        result[0][3] = (-a10*b09 + a11*b07 - a12*b06)*inv_det;
        result[1][3] = (a00*b09 - a01*b07 + a02*b06)*inv_det;
        result[2][3] = (-a30*b03 + a31*b01 - a32*b00)*inv_det;
        result[3][3] = (a20*b03 - a21*b01 + a22*b00)*inv_det;

        result
    }

    pub fn identity() -> Self {
        mat4![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    }

    pub fn translation(v: Vec3) -> Self {
        mat4![1.0, 0.0, 0.0, v[0],
              0.0, 1.0, 0.0, v[1],
              0.0, 0.0, 1.0, v[2],
              0.0, 0.0, 0.0, 1.0]
    }

    pub fn rotation(angle: f32, axis: Vec3) -> Self {
        let invnorm = 1.0/Vec3::norm(axis);
        let x = axis[0] * invnorm;
        let y = axis[1] * invnorm;
        let z = axis[2] * invnorm;
        let c = f32::cos(angle);
        let s = f32::sin(angle);
        let t = 1.0 - c;

        mat4![
            t*x*x + c, t*x*y - z*s, t*x*z + y*s, 0.0,
            t*x*y + z*s, t*y*y + c, t*y*z - x*s, 0.0,
            t*x*z - y*s, t*y*z + x*s, t*z*z + c, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    }

    pub fn scaling(v: Vec3) -> Self {
        mat4![
            v[0], 0.0, 0.0, 0.0,
            0.0, v[1], 0.0, 0.0,
            0.0, 0.0, v[2], 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    }
}

impl FromIterator<f32> for Mat4 {
    fn from_iter<I: IntoIterator<Item=f32>>(iter: I) -> Self {
        let mut mat = Mat4::zero();
        let mut index = 0usize;
        for x in iter {
            mat.elems[index] = x;
            index += 1;
        }
        mat
    }
}

impl ops::Index<usize> for Mat4 {
    type Output = [f32];
    fn index(&self, i: usize) -> &Self::Output {
        &self.elems[4*i .. (4*i)+4]
    }
}

impl ops::IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.elems[4*i .. (4*i)+4]
    }
}

impl ops::Add for Mat4 {
    type Output = Self;
    fn add(self, v: Self) -> Self {
        Mat4::binop(self, v, |x, y| x + y)
    }
}

impl ops::Sub for Mat4 {
    type Output = Self;
    fn sub(self, v: Self) -> Self {
        Mat4::binop(self, v, |x, y| x - y)
    }
}

impl ops::Mul<f32> for Mat4 {
    type Output = Self;
    fn mul(self, a: f32) -> Self {
        Mat4::unop(self, |x| x * a)
    }
}

impl ops::Mul<Mat4> for f32 {
    type Output = Mat4;
    fn mul(self, m: Self::Output) -> Self::Output {
        m * self
    }
}

impl ops::Mul<Mat4> for Mat4 {
    type Output = Self;
    fn mul(self, m: Self) -> Self {
        let mut ret = Self::zero();
        for col in 0..4 {
            for row in 0..4 {
                let mut dp: f32 = 0.0;
                for k in 0..4 {
                    dp += self[row][k] * m[k][col];
                }
                ret[row][col] = dp;
            }
        }
        ret
    }
}

impl ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Self::Output) -> Self::Output {
        let mut ret = Self::Output::zero();
        for row in 0..4 {
            let mut dp: f32 = 0.0;
            for col in 0..4 {
                dp += self[row][col] * v[col];
            }
            ret[row] = dp;
        }
        ret
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_transpose() {
        let a = mat4![
            0.923812,  0.844113, 0.833776,  0.659936,
            0.0233313, 0.708602, 0.0627441, 0.473035,
            0.371458,  0.132694, 0.359072,  0.500714,
            0.0634159, 0.504204, 0.368719,  0.562381
        ];

        let tr = mat4![
            0.923812, 0.0233313, 0.371458, 0.0634159,
            0.844113, 0.708602,  0.132694, 0.504204,
            0.833776, 0.0627441, 0.359072, 0.368719,
            0.659936, 0.473035,  0.500714, 0.562381
        ];

        assert!(Mat4::dist(Mat4::transpose(a), tr) < 1e-3);
    }

    #[test]
    fn matrix_multiplication() {
        let a = mat4![
             0.407948, 0.391539, 0.49619,  0.322796,
             0.858607, 0.545399, 0.735086, 0.427642,
             0.167469, 0.243623, 0.767047, 0.438374,
             0.327309, 0.354291, 0.872581, 0.323766
        ];

        let b = mat4![
            0.351536, 0.849369,  0.378837, 0.612026,
            0.82537,  0.431211,  0.116542, 0.175911,
            0.431053, 0.0674252, 0.531087, 0.629576,
            0.637198, 0.908619,  0.448954, 0.588895
        ];

        let correct_result = mat4![
            0.886142, 0.842089, 0.608617, 0.821033,
            1.34134,  1.40258,  0.97122,  1.33606,
            0.86992,  0.69733,  0.696015, 0.886423,
            0.989913, 0.783794, 0.774059, 1.00266
        ];

        let result = a * b;
        assert!(Mat4::dist(result, correct_result) < 1e-3);
    }

    #[test]
    fn matrix_tranform_vec4() {
        let a = mat4! [
            0.117138,   0.051085,  0.586223,  0.202724,
            0.51642,    0.681734,  0.675875,  0.121768,
            0.0954671,  0.477204,  0.450969,  0.349518,
            0.419533,   0.118843,  0.940717,  0.459807
        ];

        let x = vec4![
            0.940325,  0.915541,  0.160949,  0.297068
        ];

        let correct = vec4![
            0.311493,  1.25471,  0.703083,  0.791304
        ];

        let result = a * x;
        assert!(Vec4::dist(result, correct) < 1e-3);
    }

    #[test]
    fn matrix_inverse() {
        let a = mat4![
            0.104246, 0.415759, 0.281168, 0.393078,
            0.499819, 0.916653, 0.8353,   0.287967,
            0.403391, 0.49504,  0.963825, 0.308903,
            0.250295, 0.853082, 0.277671, 0.71118
        ];

        let correct_inv = mat4![
            -18.085,   -2.90467,   5.22884,   8.90076,
            5.67878,  3.06495,  -3.48748,  -2.86498,
            5.48158,  0.562537, -0.126435, -3.2026,
            -2.58719, -2.87386,   2.39244,   2.96059
        ];

        let result = Mat4::inverse(a);
        assert!(Mat4::dist(result, correct_inv) < 1e-3);
    }
}

// geometry
// ============================================================================


pub fn face_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    return Vec3::normalize(Vec3::cross(e1, e2));
}

pub fn overlap(min1: f32, max1: f32, min2: f32, max2: f32) -> bool {
    f32::max(min1, min2) <= f32::min(max1, max2)
}

#[derive(Clone, Copy)]
pub struct Triangle {
    pub p: [Vec3; 3]
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Triangle{p: [a, b, c]}
    }

    pub fn normal(&self) -> Vec3 {
        let e1 = self.p[1] - self.p[0];
        let e2 = self.p[2] - self.p[0];
        Vec3::cross(e1, e2)
    }

    pub fn unit_normal(&self) -> Vec3 {
        Vec3::normalize(self.normal())
    }

    pub fn project_to_axis(&self, axis: Vec3) -> (f32, f32) {
        let mut min = Vec3::dot(self.p[0], axis);
        let mut max = min;
        for i in 1..3 {
            let proj = Vec3::dot(self.p[i], axis);
            min = f32::min(min, proj);
            max = f32::max(max, proj);
        }
        (min, max)
    }

    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let boxnormals = [vec3![1.0, 0.0, 0.0], vec3![0.0, 1.0, 0.0], vec3![0.0, 0.0, 1.0]];
        for i in 0..3 {
            let (trimin, trimax) = self.project_to_axis(boxnormals[i]);
            if trimax < aabb.min[i] || trimin > aabb.max[i] {
                return false;
            }
        }
        let trinormal = self.normal();
        let trioffset = Vec3::dot(trinormal, self.p[0]);
        let (boxmin, boxmax) = aabb.project_to_axis(trinormal);
        if boxmax < trioffset || boxmin > trioffset {
            return false;
        }
        let triedges = [self.p[0]-self.p[1], self.p[1]-self.p[2], self.p[2]-self.p[0]];
        for i in 0..3 {
            for j in 0..3 {
                let axis = Vec3::cross(triedges[i], boxnormals[j]);
                let (boxmin, boxmax) = aabb.project_to_axis(axis);
                let (trimin, trimax) = self.project_to_axis(axis);
                if boxmax < trimin || boxmin > trimax {
                    return false;
                }
            }
        }
        true
    }
}


#[derive(Clone, Copy)]
pub struct AABB {
   pub min: Vec3,
   pub max: Vec3
}

impl AABB {
    pub fn max_min() -> Self {
        Self {
            min: vec3![f32::MAX, f32::MAX, f32::MAX],
            max: vec3![f32::MIN, f32::MIN, f32::MIN]
        }
    }

    pub fn merge(a: &AABB, b: &AABB) -> AABB {
        AABB{min: Vec3::min(a.min, b.min), max: Vec3::max(a.max, b.max)}
    }

    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    pub fn max_extent(&self) -> f32 {
        (self.max[0] - self.min[0])
            .max(self.max[1] - self.min[1])
            .max(self.max[2] - self.min[2])
    }

    pub fn corners(&self) -> [Vec3; 8] {
        [
            vec3![self.min[0], self.min[1], self.min[2]],
            vec3![self.max[0], self.min[1], self.min[2]],
            vec3![self.max[0], self.min[1], self.max[2]],
            vec3![self.min[0], self.min[1], self.max[2]],
            vec3![self.min[0], self.max[1], self.min[2]],
            vec3![self.max[0], self.max[1], self.min[2]],
            vec3![self.max[0], self.max[1], self.max[2]],
            vec3![self.min[0], self.max[1], self.max[2]],
        ]
    }

    pub fn corners_vec4(&self) -> [Vec4; 8] {
        [
            vec4![self.min[0], self.min[1], self.min[2], 1.0],
            vec4![self.max[0], self.min[1], self.min[2], 1.0],
            vec4![self.max[0], self.min[1], self.max[2], 1.0],
            vec4![self.min[0], self.min[1], self.max[2], 1.0],
            vec4![self.min[0], self.max[1], self.min[2], 1.0],
            vec4![self.max[0], self.max[1], self.min[2], 1.0],
            vec4![self.max[0], self.max[1], self.max[2], 1.0],
            vec4![self.min[0], self.max[1], self.max[2], 1.0],
        ]
    }

    pub fn contains(&self, v: Vec3) -> bool {
        self.min[0] <= v[0] && v[0] <= self.max[0] &&
        self.min[1] <= v[1] && v[1] <= self.max[1] &&
        self.min[2] <= v[2] && v[2] <= self.max[2]
    }

    pub fn contains_vec4(&self, v: Vec4) -> bool {
        self.min[0] <= v[0] && v[0] <= self.max[0] &&
        self.min[1] <= v[1] && v[1] <= self.max[1] &&
        self.min[2] <= v[2] && v[2] <= self.max[2]
    }

    pub fn project_to_axis(&self, axis: Vec3) -> (f32, f32) {
        let verts = self.corners();
        let mut min = Vec3::dot(verts[0], axis);
        let mut max = min;
        for i in 1..8 {
            let proj = Vec3::dot(verts[i], axis);
            min = f32::min(min, proj);
            max = f32::max(max, proj);
        }
        (min, max)
    }
}

pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3
}

impl Plane {
    pub fn new() -> Self {
        Plane {point: Vec3::zero(), normal: Vec3::zero()}
    }

    pub fn signed_dist(plane: &Plane, point: Vec3) -> f32 {
        Vec3::dot(plane.normal, point - plane.point)
    }

    pub fn signed_dist_vec4(plane: &Plane, point: Vec4) -> f32 {
        Plane::signed_dist(plane, point.to_vec3())
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3
}

impl Ray {
    pub fn intersects_triangle(&self, tri: &Triangle) -> Option<(Vec3, Vec3)> {
        const EPS: f32 = 1e-8;
        let v0v1 = tri.p[1] - tri.p[0];
        let v0v2 = tri.p[2] - tri.p[0];
        let n = Vec3::cross(v0v1, v0v2);
        let denom = Vec3::dot(n, n);
        let ndotraydir = Vec3::dot(n, self.dir);
        if f32::abs(ndotraydir) < EPS { return None; } // ray is parallel to triangle
        let d = -Vec3::dot(n, tri.p[0]);
        let t = -(Vec3::dot(n, self.origin) + d) / ndotraydir;
        if t < 0.0 { return None; } // triangle is behind the origin
        let ipoint = self.origin + t*self.dir; // intersection point of ray and plane
                                             
        let v1p = ipoint - tri.p[1];
        let v1v2 = tri.p[2] - tri.p[1];
        let mut c = Vec3::cross(v1v2, v1p);
        let mut u = Vec3::dot(n, c);
        if u < 0.0 { return None; }

        let v2p = ipoint - tri.p[2];
        let v2v0 = tri.p[0] - tri.p[2];
        c = Vec3::cross(v2v0, v2p);
        let mut v = Vec3::dot(n, c);
        if v < 0.0 { return None; }

        let v0p = ipoint - tri.p[0];
        c = Vec3::cross(v0v1, v0p);
        if Vec3::dot(n, c) < 0.0 { return None; }

        u /= denom;
        v /= denom;
        let w = 1.0 - u - v;
        Some((ipoint, vec3![u, v, w]))
    }

    pub fn intersects_aabb(&self, aabb: &AABB) -> Option<Vec3> {
        const RIGHT: u8 = 0;
        const LEFT: u8 = 1;
        const MIDDLE: u8 = 2;

        let mut inside = true;
        let mut quadrant = [RIGHT; 3];
        let mut maxt = Vec3::zero();
        let mut cand_plane = Vec3::zero();

        for i in 0..3 {
            if self.origin[i] < aabb.min[i] {
                quadrant[i] = LEFT;
                cand_plane[i] = aabb.min[i];
                inside = false;
            } else if self.origin[i] > aabb.max[i] {
                quadrant[i] = RIGHT;
                cand_plane[i] = aabb.max[i];
                inside = false;
            } else {
                quadrant[i] = MIDDLE;
            }
        }

        if inside {
            return Some(self.origin);
        }

        for i in 0..3 {
            if quadrant[i] != MIDDLE && self.dir[i] != 0.0 {
                maxt[i] = (cand_plane[i] - self.origin[i]) / self.dir[i];
            } else {
                maxt[i] = -1.0;
            }
        }

        let mut which_plane = 0usize;
        for i in 1..3 {
            if maxt[which_plane] < maxt[i] {
                which_plane = i;
            }
        }

        if maxt[which_plane] < 0.0 { return None; }
        let mut ip = Vec3::zero();
        for i in 0..3 {
            if which_plane != i {
                ip[i] = self.origin[i] + maxt[which_plane]*self.dir[i];
                if ip[i] < aabb.min[i] || ip[i] > aabb.max[i] {
                    return None;
                }
            } else {
                ip[i] = cand_plane[i];
            }
        }
        Some(ip)
    }
}

