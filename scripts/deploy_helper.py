#!/usr/bin/env python3
"""
Script d'aide pour le déploiement
Aide à gérer les versions, tags, et déploiements
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Execute a shell command"""
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=capture_output, 
        text=True
    )
    return result


def get_current_version():
    """Get current git tag/version"""
    result = run_command("git describe --tags --abbrev=0")
    if result.returncode == 0:
        return result.stdout.strip()
    return "v0.0.0"


def create_tag(version, message):
    """Create a new git tag"""
    print(f"Creating tag {version}...")
    
    # Create tag
    result = run_command(f'git tag -a {version} -m "{message}"')
    if result.returncode != 0:
        print(f"❌ Failed to create tag: {result.stderr}")
        return False
    
    print(f"✓ Tag {version} created")
    
    # Push tag
    print("Pushing tag to remote...")
    result = run_command(f"git push origin {version}")
    if result.returncode != 0:
        print(f"❌ Failed to push tag: {result.stderr}")
        return False
    
    print(f"✓ Tag {version} pushed to remote")
    print("\n🚀 CI/CD pipeline will be triggered automatically!")
    return True


def list_docker_images():
    """List available Docker images"""
    print("\n📦 Docker Images:")
    print("-" * 60)
    
    # This would query your container registry
    print("\nTo list images from GitHub Container Registry:")
    print("1. Go to: https://github.com/orgs/YOUR_ORG/packages")
    print("2. Or use: gh api /user/packages?package_type=container")


def deploy_local(version="latest"):
    """Deploy locally using docker-compose"""
    print(f"\n🚀 Deploying version {version} locally...")
    
    # Check if docker-compose.yml exists
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return False
    
    # Pull images
    print("\nPulling Docker images...")
    result = run_command("docker-compose pull", capture_output=False)
    if result.returncode != 0:
        print("❌ Failed to pull images")
        return False
    
    # Start services
    print("\nStarting services...")
    result = run_command("docker-compose up -d", capture_output=False)
    if result.returncode != 0:
        print("❌ Failed to start services")
        return False
    
    print("\n✓ Services started successfully!")
    print("\nAccess the API at: http://localhost:8000")
    print("Access API docs at: http://localhost:8000/docs")
    return True


def stop_local():
    """Stop local deployment"""
    print("\n🛑 Stopping local deployment...")
    result = run_command("docker-compose down", capture_output=False)
    if result.returncode == 0:
        print("✓ Services stopped")
        return True
    else:
        print("❌ Failed to stop services")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for MLOps deployment"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show current version')
    
    # Tag command
    tag_parser = subparsers.add_parser('tag', help='Create a new version tag')
    tag_parser.add_argument('version', help='Version tag (e.g., v1.0.0)')
    tag_parser.add_argument('-m', '--message', default='Release', help='Tag message')
    
    # Images command
    images_parser = subparsers.add_parser('images', help='List Docker images')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy locally')
    deploy_parser.add_argument('--version', default='latest', help='Version to deploy')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop local deployment')
    
    args = parser.parse_args()
    
    if args.command == 'version':
        version = get_current_version()
        print(f"Current version: {version}")
    
    elif args.command == 'tag':
        if not args.version.startswith('v'):
            print("❌ Version must start with 'v' (e.g., v1.0.0)")
            sys.exit(1)
        
        create_tag(args.version, args.message)
    
    elif args.command == 'images':
        list_docker_images()
    
    elif args.command == 'deploy':
        deploy_local(args.version)
    
    elif args.command == 'stop':
        stop_local()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
