use rand::Rng;

use crate::hittable::HitRecord;
use crate::ray::{Color, Ray, Vec3};

pub enum Material {
    Lambertian { albedo: Color },
    Metal { albedo: Color, fuzz: f64 },
    Dielectric { ior: f64 },
}

pub struct ScatterResult {
    pub attenuation: Color,
    pub scattered: Ray,
}

impl Material {
    pub fn scatter(&self, ray_in: &Ray, rec: &HitRecord, rng: &mut impl Rng) -> Option<ScatterResult> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_dir = rec.normal + Vec3::random_unit_vector(rng);
                if scatter_dir.near_zero() {
                    scatter_dir = rec.normal;
                }
                Some(ScatterResult {
                    attenuation: *albedo,
                    scattered: Ray::new(rec.point, scatter_dir),
                })
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = ray_in.direction.unit().reflect(rec.normal);
                let scattered = Ray::new(
                    rec.point,
                    reflected + Vec3::random_in_unit_sphere(rng) * *fuzz,
                );
                if scattered.direction.dot(rec.normal) > 0.0 {
                    Some(ScatterResult {
                        attenuation: *albedo,
                        scattered,
                    })
                } else {
                    None
                }
            }
            Material::Dielectric { ior } => {
                let refraction_ratio = if rec.front_face { 1.0 / ior } else { *ior };
                let unit_direction = ray_in.direction.unit();
                let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let direction = if cannot_refract
                    || reflectance(cos_theta, refraction_ratio) > rng.gen()
                {
                    unit_direction.reflect(rec.normal)
                } else {
                    unit_direction.refract(rec.normal, refraction_ratio)
                };

                Some(ScatterResult {
                    attenuation: Color::new(1.0, 1.0, 1.0),
                    scattered: Ray::new(rec.point, direction),
                })
            }
        }
    }
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Schlick's approximation
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::{Point3, Vec3};

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn make_hit_record(
        point: Point3,
        normal: Vec3,
        t: f64,
        front_face: bool,
        material: &Material,
    ) -> HitRecord {
        HitRecord {
            point,
            normal,
            t,
            front_face,
            material,
        }
    }

    // ---- Reflectance (Schlick's approximation) ----

    #[test]
    fn reflectance_at_normal_incidence() {
        // At normal incidence (cosine = 1), reflectance = r0
        let ref_idx: f64 = 1.5;
        let r0 = ((1.0_f64 - ref_idx) / (1.0_f64 + ref_idx)).powi(2);
        assert!(approx_eq(reflectance(1.0, ref_idx), r0));
    }

    #[test]
    fn reflectance_at_grazing_angle() {
        // At grazing angle (cosine = 0), reflectance = 1.0
        assert!(approx_eq(reflectance(0.0, 1.5), 1.0));
    }

    #[test]
    fn reflectance_monotonic() {
        // Reflectance should increase as angle increases (cosine decreases)
        let ref_idx = 1.5;
        let r1 = reflectance(0.9, ref_idx);
        let r2 = reflectance(0.5, ref_idx);
        let r3 = reflectance(0.1, ref_idx);
        assert!(r1 < r2);
        assert!(r2 < r3);
    }

    #[test]
    fn reflectance_between_zero_and_one() {
        let ref_idx = 1.5;
        for i in 0..=10 {
            let cosine = i as f64 / 10.0;
            let r = reflectance(cosine, ref_idx);
            assert!(r >= 0.0 && r <= 1.0, "reflectance({}) = {} out of range", cosine, r);
        }
    }

    // ---- Lambertian scattering ----

    #[test]
    fn lambertian_always_scatters() {
        let mut rng = rand::thread_rng();
        let material = Material::Lambertian {
            albedo: Color::new(0.8, 0.3, 0.1),
        };
        let ray_in = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        for _ in 0..50 {
            let result = material.scatter(&ray_in, &rec, &mut rng);
            assert!(result.is_some());
        }
    }

    #[test]
    fn lambertian_attenuation_matches_albedo() {
        let mut rng = rand::thread_rng();
        let albedo = Color::new(0.8, 0.3, 0.1);
        let material = Material::Lambertian { albedo };
        let ray_in = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        assert!(approx_eq(result.attenuation.x, albedo.x));
        assert!(approx_eq(result.attenuation.y, albedo.y));
        assert!(approx_eq(result.attenuation.z, albedo.z));
    }

    #[test]
    fn lambertian_scattered_ray_originates_at_hit_point() {
        let mut rng = rand::thread_rng();
        let material = Material::Lambertian {
            albedo: Color::new(0.5, 0.5, 0.5),
        };
        let hit_point = Point3::new(1.0, 2.0, 3.0);
        let ray_in = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = make_hit_record(
            hit_point,
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        assert!(approx_eq(result.scattered.origin.x, hit_point.x));
        assert!(approx_eq(result.scattered.origin.y, hit_point.y));
        assert!(approx_eq(result.scattered.origin.z, hit_point.z));
    }

    // ---- Metal scattering ----

    #[test]
    fn metal_reflects_ray() {
        let mut rng = rand::thread_rng();
        let material = Material::Metal {
            albedo: Color::new(0.8, 0.8, 0.8),
            fuzz: 0.0,
        };
        let ray_in = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0).unit(),
        );
        let rec = make_hit_record(
            Point3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        // With zero fuzz, reflected direction should be (1,1,0).unit()
        let expected_dir = Vec3::new(1.0, 1.0, 0.0).unit();
        assert!(approx_eq(result.scattered.direction.unit().x, expected_dir.x));
        assert!(approx_eq(result.scattered.direction.unit().y, expected_dir.y));
    }

    #[test]
    fn metal_zero_fuzz_perfect_reflection() {
        let mut rng = rand::thread_rng();
        let material = Material::Metal {
            albedo: Color::new(1.0, 1.0, 1.0),
            fuzz: 0.0,
        };
        // Ray going straight down
        let ray_in = Ray::new(
            Point3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        // Perfect reflection straight up
        let dir = result.scattered.direction.unit();
        assert!(approx_eq(dir.x, 0.0));
        assert!(approx_eq(dir.y, 1.0));
        assert!(approx_eq(dir.z, 0.0));
    }

    #[test]
    fn metal_attenuation_matches_albedo() {
        let mut rng = rand::thread_rng();
        let albedo = Color::new(0.7, 0.3, 0.9);
        let material = Material::Metal { albedo, fuzz: 0.0 };
        let ray_in = Ray::new(
            Point3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        assert!(approx_eq(result.attenuation.x, albedo.x));
        assert!(approx_eq(result.attenuation.y, albedo.y));
        assert!(approx_eq(result.attenuation.z, albedo.z));
    }

    #[test]
    fn metal_fuzz_adds_randomness() {
        let mut rng = rand::thread_rng();
        let material = Material::Metal {
            albedo: Color::new(1.0, 1.0, 1.0),
            fuzz: 0.5,
        };
        let ray_in = Ray::new(
            Point3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            5.0,
            true,
            &material,
        );

        // Collect many scattered directions -- with fuzz, they should vary
        let mut directions = Vec::new();
        for _ in 0..20 {
            if let Some(result) = material.scatter(&ray_in, &rec, &mut rng) {
                directions.push(result.scattered.direction);
            }
        }
        // At least some directions should differ from perfect reflection
        let perfect = Vec3::new(0.0, 1.0, 0.0);
        let deviations: Vec<f64> = directions
            .iter()
            .map(|d| (d.unit() - perfect).length())
            .collect();
        let any_deviated = deviations.iter().any(|&d| d > 0.01);
        assert!(any_deviated, "Fuzzy metal should produce varied reflections");
    }

    // ---- Dielectric scattering ----

    #[test]
    fn dielectric_always_scatters() {
        let mut rng = rand::thread_rng();
        let material = Material::Dielectric { ior: 1.5 };
        let ray_in = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        for _ in 0..50 {
            let result = material.scatter(&ray_in, &rec, &mut rng);
            assert!(result.is_some());
        }
    }

    #[test]
    fn dielectric_attenuation_is_white() {
        let mut rng = rand::thread_rng();
        let material = Material::Dielectric { ior: 1.5 };
        let ray_in = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        assert!(approx_eq(result.attenuation.x, 1.0));
        assert!(approx_eq(result.attenuation.y, 1.0));
        assert!(approx_eq(result.attenuation.z, 1.0));
    }

    #[test]
    fn dielectric_scattered_ray_originates_at_hit_point() {
        let mut rng = rand::thread_rng();
        let material = Material::Dielectric { ior: 1.5 };
        let hit_point = Point3::new(2.0, 3.0, -5.0);
        let ray_in = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        let rec = make_hit_record(
            hit_point,
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            true,
            &material,
        );

        let result = material.scatter(&ray_in, &rec, &mut rng).unwrap();
        assert!(approx_eq(result.scattered.origin.x, hit_point.x));
        assert!(approx_eq(result.scattered.origin.y, hit_point.y));
        assert!(approx_eq(result.scattered.origin.z, hit_point.z));
    }

    #[test]
    fn dielectric_total_internal_reflection() {
        // When going from high IOR to low IOR at steep angle, should get total internal reflection
        let mut rng = rand::thread_rng();
        let material = Material::Dielectric { ior: 2.5 };
        // Steep angle -- coming from inside the material
        let ray_in = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.9, 0.0, -0.1).unit(),
        );
        let rec = make_hit_record(
            Point3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            false, // back face -- inside the material
            &material,
        );

        // Should still scatter (either reflect or refract)
        let result = material.scatter(&ray_in, &rec, &mut rng);
        assert!(result.is_some());
    }
}
