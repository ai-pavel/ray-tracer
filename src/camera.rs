use rand::Rng;

use crate::ray::{Point3, Ray, Vec3};

pub struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    pub fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov_deg: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov_deg.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).unit();
        let u = vup.cross(w).unit();
        let v = w.cross(u);

        let origin = lookfrom;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    pub fn get_ray(&self, s: f64, t: f64, rng: &mut impl Rng) -> Ray {
        let rd = Vec3::random_in_unit_disk(rng) * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + self.horizontal * s + self.vertical * t
                - self.origin
                - offset,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn default_camera() -> Camera {
        Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            90.0,
            16.0 / 9.0,
            0.0, // no aperture => no DOF blur
            1.0,
        )
    }

    #[test]
    fn camera_ray_origin_no_aperture() {
        let cam = default_camera();
        let mut rng = rand::thread_rng();
        let ray = cam.get_ray(0.5, 0.5, &mut rng);
        // With zero aperture, origin should always be the camera origin
        assert!(approx_eq(ray.origin.x, 0.0));
        assert!(approx_eq(ray.origin.y, 0.0));
        assert!(approx_eq(ray.origin.z, 0.0));
    }

    #[test]
    fn camera_center_ray_points_forward() {
        let cam = default_camera();
        let mut rng = rand::thread_rng();
        let ray = cam.get_ray(0.5, 0.5, &mut rng);
        // Center ray should point roughly toward -z
        let dir = ray.direction.unit();
        assert!(dir.z < 0.0, "Center ray should point in -z direction");
        // x and y should be near zero for center
        assert!(dir.x.abs() < 0.1);
        assert!(dir.y.abs() < 0.1);
    }

    #[test]
    fn camera_corner_rays_diverge() {
        let cam = default_camera();
        let mut rng = rand::thread_rng();
        let top_left = cam.get_ray(0.0, 1.0, &mut rng);
        let bottom_right = cam.get_ray(1.0, 0.0, &mut rng);

        // Rays in opposite corners should diverge
        let tl_dir = top_left.direction.unit();
        let br_dir = bottom_right.direction.unit();
        // They should be different
        assert!((tl_dir - br_dir).length() > 0.1);
    }

    #[test]
    fn camera_with_aperture_varies_origin() {
        let cam = Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            90.0,
            16.0 / 9.0,
            2.0, // large aperture
            1.0,
        );
        let mut rng = rand::thread_rng();

        let mut origins = Vec::new();
        for _ in 0..20 {
            let ray = cam.get_ray(0.5, 0.5, &mut rng);
            origins.push(ray.origin);
        }

        // With a large aperture, origins should vary
        let any_different = origins
            .windows(2)
            .any(|w| (w[0] - w[1]).length() > 0.001);
        assert!(any_different, "Aperture should cause varied ray origins");
    }

    #[test]
    fn camera_different_fov() {
        let narrow = Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            20.0, // narrow FOV
            1.0,
            0.0,
            1.0,
        );
        let wide = Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            120.0, // wide FOV
            1.0,
            0.0,
            1.0,
        );
        let mut rng = rand::thread_rng();

        // Corner ray for narrow FOV
        let narrow_corner = narrow.get_ray(1.0, 1.0, &mut rng).direction.unit();
        // Corner ray for wide FOV
        let wide_corner = wide.get_ray(1.0, 1.0, &mut rng).direction.unit();

        // Wide FOV corner ray should have more divergence from center (-z)
        let narrow_angle = narrow_corner.dot(Vec3::new(0.0, 0.0, -1.0)).acos();
        let wide_angle = wide_corner.dot(Vec3::new(0.0, 0.0, -1.0)).acos();
        assert!(
            wide_angle > narrow_angle,
            "Wide FOV should produce larger angle from center"
        );
    }
}
