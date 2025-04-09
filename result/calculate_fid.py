from pytorch_fid import fid_score

if __name__ == '__main__':
    real_image_dir = './real_T1'
    fake_image_dir = './fake_T1'
    fid_value = fid_score.calculate_fid_given_paths([real_image_dir, fake_image_dir], batch_size=50, device="cuda", dims=2048)
    print(f"T1 FID Score: {fid_value}")

    real_image_dir = './real_T2'
    fake_image_dir = './fake_T2'
    fid_value = fid_score.calculate_fid_given_paths([real_image_dir, fake_image_dir], batch_size=50, device="cuda", dims=2048)
    print(f"T2 FID Score: {fid_value}")