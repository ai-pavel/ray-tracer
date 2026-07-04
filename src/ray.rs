use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub};

use rand::Rng;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn unit(self) -> Self {
        let len = self.length();
        self / len
    }

    pub fn near_zero(self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    pub fn reflect(self, n: Self) -> Self {
        self - n * 2.0 * self.dot(n)
    }

    pub fn refract(self, n: Self, etai_over_etat: f64) -> Self {
        let cos_theta = (-self).dot(n).min(1.0);
        let r_out_perp = (self + n * cos_theta) * etai_over_etat;
        let r_out_parallel = n * -(1.0 - r_out_perp.length_squared()).abs().sqrt();
        r_out_perp + r_out_parallel
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        Self::new(rng.gen(), rng.gen(), rng.gen())
    }

    pub fn random_range(rng: &mut impl Rng, min: f64, max: f64) -> Self {
        Self::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    pub fn random_in_unit_sphere(rng: &mut impl Rng) -> Self {
        loop {
            let p = Self::random_range(rng, -1.0, 1.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_unit_vector(rng: &mut impl Rng) -> Self {
        Self::random_in_unit_sphere(rng).unit()
    }

    pub fn random_in_unit_disk(rng: &mut impl Rng) -> Self {
        loop {
            let p = Self::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn min_components(a: Self, b: Self) -> Self {
        Self::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }

    pub fn max_components(a: Self, b: Self) -> Self {
        Self::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }

    pub fn component(self, index: usize) -> f64 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

// Operator overloads

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, t: f64) -> Self {
        Self::new(self.x * t, self.y * t, self.z * t)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        *self = *self * t;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, t: f64) -> Self {
        self * (1.0 / t)
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        *self = *self / t;
    }
}

pub type Color = Vec3;
pub type Point3 = Vec3;

// Ray

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

// AABB for BVH

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn hit(&self, ray: &Ray, mut t_min: f64, mut t_max: f64) -> bool {
        for a in 0..3 {
            let inv_d = 1.0 / ray.direction.component(a);
            let mut t0 = (self.min.component(a) - ray.origin.component(a)) * inv_d;
            let mut t1 = (self.max.component(a) - ray.origin.component(a)) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = t0.max(t_min);
            t_max = t1.min(t_max);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }

    pub fn surrounding_box(a: Aabb, b: Aabb) -> Aabb {
        Aabb {
            min: Vec3::min_components(a.min, b.min),
            max: Vec3::max_components(a.max, b.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // ---- Vec3 construction and basic properties ----

    #[test]
    fn vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn vec3_default_is_zero() {
        let v = Vec3::default();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vec3_clone_and_copy() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let v2 = v; // Copy
        let v3 = v.clone(); // Clone
        assert!(vec3_approx_eq(v, v2));
        assert!(vec3_approx_eq(v, v3));
    }

    // ---- Vec3 length operations ----

    #[test]
    fn vec3_length_squared() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(approx_eq(v.length_squared(), 14.0));
    }

    #[test]
    fn vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx_eq(v.length(), 5.0));
    }

    #[test]
    fn vec3_length_unit_vector() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert!(approx_eq(v.length(), 1.0));
    }

    #[test]
    fn vec3_length_zero_vector() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        assert!(approx_eq(v.length(), 0.0));
    }

    // ---- Dot product ----

    #[test]
    fn vec3_dot_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx_eq(a.dot(b), 32.0)); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn vec3_dot_product_perpendicular() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx_eq(a.dot(b), 0.0));
    }

    #[test]
    fn vec3_dot_product_parallel() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        assert!(approx_eq(a.dot(b), 6.0));
    }

    #[test]
    fn vec3_dot_product_antiparallel() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(-1.0, 0.0, 0.0);
        assert!(approx_eq(a.dot(b), -1.0));
    }

    // ---- Cross product ----

    #[test]
    fn vec3_cross_product_basis_vectors() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let j = Vec3::new(0.0, 1.0, 0.0);
        let k = Vec3::new(0.0, 0.0, 1.0);

        assert!(vec3_approx_eq(i.cross(j), k));
        assert!(vec3_approx_eq(j.cross(k), i));
        assert!(vec3_approx_eq(k.cross(i), j));
    }

    #[test]
    fn vec3_cross_product_anticommutative() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a.cross(b), -b.cross(a)));
    }

    #[test]
    fn vec3_cross_product_parallel_is_zero() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let cross = a.cross(b);
        assert!(approx_eq(cross.length(), 0.0));
    }

    #[test]
    fn vec3_cross_product_perpendicular_to_inputs() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a.cross(b);
        assert!(approx_eq(c.dot(a), 0.0));
        assert!(approx_eq(c.dot(b), 0.0));
    }

    // ---- Unit vector / normalization ----

    #[test]
    fn vec3_unit_vector() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let u = v.unit();
        assert!(approx_eq(u.length(), 1.0));
        assert!(approx_eq(u.x, 0.6));
        assert!(approx_eq(u.y, 0.8));
        assert!(approx_eq(u.z, 0.0));
    }

    #[test]
    fn vec3_unit_preserves_direction() {
        let v = Vec3::new(5.0, 0.0, 0.0);
        let u = v.unit();
        assert!(vec3_approx_eq(u, Vec3::new(1.0, 0.0, 0.0)));
    }

    // ---- near_zero ----

    #[test]
    fn vec3_near_zero_true() {
        let v = Vec3::new(1e-9, -1e-9, 1e-10);
        assert!(v.near_zero());
    }

    #[test]
    fn vec3_near_zero_false() {
        let v = Vec3::new(0.1, 0.0, 0.0);
        assert!(!v.near_zero());
    }

    // ---- Reflection ----

    #[test]
    fn vec3_reflect_horizontal_surface() {
        // Ray going down-right hitting a horizontal surface (normal pointing up)
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let reflected = v.reflect(n);
        assert!(vec3_approx_eq(reflected, Vec3::new(1.0, 1.0, 0.0)));
    }

    #[test]
    fn vec3_reflect_vertical_surface() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let n = Vec3::new(-1.0, 0.0, 0.0);
        let reflected = v.reflect(n);
        assert!(vec3_approx_eq(reflected, Vec3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn vec3_reflect_45_degrees() {
        let v = Vec3::new(1.0, -1.0, 0.0).unit();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let reflected = v.reflect(n);
        let expected = Vec3::new(1.0, 1.0, 0.0).unit();
        assert!(vec3_approx_eq(reflected, expected));
    }

    // ---- Refraction ----

    #[test]
    fn vec3_refract_straight_through() {
        // Perpendicular incidence -- no bending regardless of IOR
        let v = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let refracted = v.refract(n, 1.5);
        assert!(vec3_approx_eq(refracted, Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn vec3_refract_same_medium() {
        // Same medium (ratio = 1.0): no bending
        let v = Vec3::new(1.0, -1.0, 0.0).unit();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let refracted = v.refract(n, 1.0);
        assert!(vec3_approx_eq(refracted, v));
    }

    #[test]
    fn vec3_refract_snells_law() {
        // Verify Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
        let angle = std::f64::consts::FRAC_PI_4; // 45 degrees
        let v = Vec3::new(angle.sin(), -angle.cos(), 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let eta = 1.0 / 1.5; // air to glass
        let refracted = v.refract(n, eta);

        let sin_theta_out = (refracted.x * refracted.x + refracted.z * refracted.z).sqrt()
            / refracted.length();
        let sin_theta_in = angle.sin();
        // n1 * sin(theta1) should equal n2 * sin(theta2), i.e., sin_theta_in = eta * sin_theta_out ... wait
        // Actually ratio = n1/n2, so sin_theta_out = ratio * sin_theta_in
        assert!((sin_theta_out - eta * sin_theta_in).abs() < 1e-6);
    }

    // ---- Operator overloads ----

    #[test]
    fn vec3_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a + b, Vec3::new(5.0, 7.0, 9.0)));
    }

    #[test]
    fn vec3_add_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a += Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a, Vec3::new(5.0, 7.0, 9.0)));
    }

    #[test]
    fn vec3_sub() {
        let a = Vec3::new(5.0, 7.0, 9.0);
        let b = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx_eq(a - b, Vec3::new(4.0, 5.0, 6.0)));
    }

    #[test]
    fn vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert!(vec3_approx_eq(-v, Vec3::new(-1.0, 2.0, -3.0)));
    }

    #[test]
    fn vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx_eq(v * 2.0, Vec3::new(2.0, 4.0, 6.0)));
    }

    #[test]
    fn vec3_scalar_mul() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec3_approx_eq(2.0 * v, Vec3::new(2.0, 4.0, 6.0)));
    }

    #[test]
    fn vec3_mul_componentwise() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(vec3_approx_eq(a * b, Vec3::new(4.0, 10.0, 18.0)));
    }

    #[test]
    fn vec3_mul_assign() {
        let mut v = Vec3::new(1.0, 2.0, 3.0);
        v *= 3.0;
        assert!(vec3_approx_eq(v, Vec3::new(3.0, 6.0, 9.0)));
    }

    #[test]
    fn vec3_div_scalar() {
        let v = Vec3::new(2.0, 4.0, 6.0);
        assert!(vec3_approx_eq(v / 2.0, Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn vec3_div_assign() {
        let mut v = Vec3::new(6.0, 9.0, 12.0);
        v /= 3.0;
        assert!(vec3_approx_eq(v, Vec3::new(2.0, 3.0, 4.0)));
    }

    // ---- Component access ----

    #[test]
    fn vec3_component_access() {
        let v = Vec3::new(10.0, 20.0, 30.0);
        assert!(approx_eq(v.component(0), 10.0));
        assert!(approx_eq(v.component(1), 20.0));
        assert!(approx_eq(v.component(2), 30.0));
    }

    #[test]
    fn vec3_component_out_of_range_returns_z() {
        let v = Vec3::new(10.0, 20.0, 30.0);
        assert!(approx_eq(v.component(99), 30.0));
    }

    // ---- Min/Max components ----

    #[test]
    fn vec3_min_components() {
        let a = Vec3::new(1.0, 5.0, 3.0);
        let b = Vec3::new(4.0, 2.0, 6.0);
        assert!(vec3_approx_eq(
            Vec3::min_components(a, b),
            Vec3::new(1.0, 2.0, 3.0)
        ));
    }

    #[test]
    fn vec3_max_components() {
        let a = Vec3::new(1.0, 5.0, 3.0);
        let b = Vec3::new(4.0, 2.0, 6.0);
        assert!(vec3_approx_eq(
            Vec3::max_components(a, b),
            Vec3::new(4.0, 5.0, 6.0)
        ));
    }

    // ---- Random vector generation ----

    #[test]
    fn vec3_random_in_unit_sphere() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let v = Vec3::random_in_unit_sphere(&mut rng);
            assert!(v.length_squared() < 1.0);
        }
    }

    #[test]
    fn vec3_random_unit_vector_is_unit_length() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let v = Vec3::random_unit_vector(&mut rng);
            assert!((v.length() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn vec3_random_in_unit_disk_has_zero_z() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let v = Vec3::random_in_unit_disk(&mut rng);
            assert!(approx_eq(v.z, 0.0));
            assert!(v.length_squared() < 1.0);
        }
    }

    #[test]
    fn vec3_random_range() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let v = Vec3::random_range(&mut rng, -5.0, 5.0);
            assert!(v.x >= -5.0 && v.x < 5.0);
            assert!(v.y >= -5.0 && v.y < 5.0);
            assert!(v.z >= -5.0 && v.z < 5.0);
        }
    }

    // ---- Ray ----

    #[test]
    fn ray_new() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        let ray = Ray::new(origin, direction);
        assert!(vec3_approx_eq(ray.origin, origin));
        assert!(vec3_approx_eq(ray.direction, direction));
    }

    #[test]
    fn ray_at_parametric() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(vec3_approx_eq(ray.at(0.0), Point3::new(0.0, 0.0, 0.0)));
        assert!(vec3_approx_eq(ray.at(1.0), Point3::new(1.0, 0.0, 0.0)));
        assert!(vec3_approx_eq(ray.at(2.5), Point3::new(2.5, 0.0, 0.0)));
        assert!(vec3_approx_eq(ray.at(-1.0), Point3::new(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn ray_at_diagonal() {
        let ray = Ray::new(
            Point3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert!(vec3_approx_eq(ray.at(2.0), Point3::new(3.0, 5.0, 7.0)));
    }

    // ---- AABB ----

    #[test]
    fn aabb_new() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(vec3_approx_eq(aabb.min, Point3::new(-1.0, -1.0, -1.0)));
        assert!(vec3_approx_eq(aabb.max, Point3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn aabb_empty() {
        let aabb = Aabb::empty();
        assert_eq!(aabb.min.x, f64::INFINITY);
        assert_eq!(aabb.max.x, f64::NEG_INFINITY);
    }

    #[test]
    fn aabb_hit_ray_through_center() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(aabb.hit(&ray, 0.0, f64::INFINITY));
    }

    #[test]
    fn aabb_miss_ray() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(!aabb.hit(&ray, 0.0, f64::INFINITY));
    }

    #[test]
    fn aabb_hit_ray_from_inside() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(aabb.hit(&ray, 0.0, f64::INFINITY));
    }

    #[test]
    fn aabb_hit_respects_t_range() {
        let aabb = Aabb::new(Point3::new(5.0, -1.0, -1.0), Point3::new(7.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Ray hits the box at t=5..7, so restricting to t=0..3 should miss
        assert!(!aabb.hit(&ray, 0.0, 3.0));
        // But t=0..10 should hit
        assert!(aabb.hit(&ray, 0.0, 10.0));
    }

    #[test]
    fn aabb_hit_negative_direction() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(aabb.hit(&ray, 0.0, f64::INFINITY));
    }

    #[test]
    fn aabb_hit_diagonal_ray() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, -5.0, -5.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(aabb.hit(&ray, 0.0, f64::INFINITY));
    }

    #[test]
    fn aabb_surrounding_box() {
        let a = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 3.0, 3.0));
        let s = Aabb::surrounding_box(a, b);
        assert!(vec3_approx_eq(s.min, Point3::new(-1.0, -1.0, -1.0)));
        assert!(vec3_approx_eq(s.max, Point3::new(3.0, 3.0, 3.0)));
    }

    #[test]
    fn aabb_surrounding_box_of_identical() {
        let a = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let s = Aabb::surrounding_box(a, a);
        assert!(vec3_approx_eq(s.min, a.min));
        assert!(vec3_approx_eq(s.max, a.max));
    }
}
